import os
import sys
import json
import re
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
from progressbar import progressbar

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.parser import get_common_parser
from tools.HTMLtraverser_factory import HTMLTraverserFactory


# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
def load_credentials(path: str):
    """Load Hugging Face credentials from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credentials file not found at: {path}")
    with open(path, "r") as f:
        creds = json.load(f)
    return creds["huggingface"]["write"]


def setup_model_and_tokenizer(classification: str, modelversion: str):
    """Load tokenizer and model for given classification/version."""
    model_name = f"gerbejon/longcoder-html-nodes-{classification}-classifier-{modelversion}"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, model_name


# ---------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------
def preprocess(node, xpath: str, modelversion: str):
    """Convert HTML node and xpath to text input for the model."""
    node = re.match(r"<[^>]+>", str(node).strip())
    node_head = node.group() if node else ""
    if modelversion == "v1":
        return f"{node_head} {xpath}"
    return None


def predict_input(text_list, tokenizer, model):
    """Predict labels for a list of input texts."""
    inputs = tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    label_mapping = model.config.id2label
    return [label_mapping[p.item()] for p in predictions]


def process_page(html: str, tokenizer, model, modelversion: str):
    """Traverse HTML and predict node-level classifications."""
    traverser = HTMLTraverserFactory.create(html, "string")
    for node, parents, dom_path, depth in traverser:
        if depth < 4:
            continue
        input_str = preprocess(node, dom_path, modelversion)
        if not input_str:
            continue
        res = predict_input([input_str], tokenizer, model)[0]
        hyu = node.attrs.get("hyu")
        if res != "multiple" and hyu is not None:
            yield {
                "tagType": res,
                "hyuIndex": int(hyu),
                "xpath": dom_path,
            }
            traverser.skip_subtree()


# ---------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------
def run_prediction(args):
    # Root paths
    rootdir = os.getcwd() if os.getcwd().split("/")[-1] != "codeBERT" else os.path.abspath(os.path.join(os.getcwd(), ".."))
    outdir = os.path.join(rootdir, f"codeBERT/results/{args.modelversion}/result_out")
    os.makedirs(outdir, exist_ok=True)

    # Login to Hugging Face
    hf_token = load_credentials(os.path.join(rootdir, "credentials.json"))
    login(hf_token)

    # Setup model
    tokenizer, model, model_name = setup_model_and_tokenizer(args.classification, args.modelversion)
    print(f"Using model: {model_name}")

    # Dataset
    dataset_name = "gerbejon/WebClasSeg25-html"
    dataset = load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name}")

    # Load label map
    id2label_path = os.path.join(rootdir, f"id2label_{args.classification}.json")
    with open(id2label_path, "r") as f:
        id2label = json.load(f)

    results_page = {}
    counter = 0
    for row in progressbar(dataset["test"]):
        counter += 1
        page_id = row["page_id"]
        html = row["html"]
        results_page[page_id] = list(process_page(html, tokenizer, model, args.modelversion))
        if counter >= 2:
            break

    # Merge with ground truth
    ann_path = os.path.join(rootdir, "data/annotations.json")
    with open(ann_path, "r") as f:
        annotations = json.load(f)

    # annotation_kind = "annotation" if args.classification == "fc" else "digilog_annotation"
    annotation_kind = f'annotation_{args.classification}'

    for item in progressbar(annotations):
        page_id = item["id"].split("_")[0]
        if page_id not in results_page:
            continue

        pred_nodes = results_page[page_id]
        gt_nodes = [{"tagType": seg["tagType"], "hyuIndex": seg["hyuIndex"]} for seg in item[annotation_kind]]

        out_page = {
            "id": page_id,
            "width": item["screenshot_size"]["width"],
            "height": item["screenshot_size"]["height"],
            "segmentations": {"predicted": [], "ground_truth": []},
            "nodes": {"predicted": pred_nodes, "ground_truth": gt_nodes},
        }

        out_page_dir = os.path.join(outdir, f"{item['id']}_codebert_{args.modelversion}")
        os.makedirs(out_page_dir, exist_ok=True)

        out_file = os.path.join(out_page_dir, f"annotations_{args.classification.upper()}.json")
        with open(out_file, "w") as f:
            json.dump(out_page, f, indent=4, ensure_ascii=False)

    print(f"Done! Predictions saved in {outdir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = get_common_parser("Run inference using fine-tuned LongCoder")
    args = parser.parse_args()

    run_prediction(args)
