from bs4 import BeautifulSoup, Tag
import re

html_tag_dict = {
    "a": "link",
    "abbr": "abbreviation",
    "address": "contact information",
    "area": "image map area",
    "article": "article section",
    "aside": "sidebar content",
    "audio": "audio player",
    "b": "bold text",
    "base": "base URL for links",
    "bdi": "bidirectional text isolate",
    "bdo": "bidirectional text override",
    "blockquote": "quotation block",
    "body": "page body",
    "br": "line break",
    "button": "clickable button",
    "canvas": "drawing area",
    "caption": "table caption",
    "cite": "citation",
    "code": "inline code snippet",
    "col": "table column",
    "colgroup": "group of table columns",
    "data": "machine-readable data",
    "datalist": "list of options",
    "dd": "description details",
    "del": "deleted text",
    "details": "expandable disclosure widget",
    "dfn": "definition term",
    "dialog": "dialog box",
    "div": "generic container",
    "dl": "description list",
    "dt": "description term",
    "em": "emphasized text",
    "embed": "embedded content",
    "fieldset": "form field group",
    "figcaption": "figure caption",
    "figure": "self-contained figure",
    "footer": "page footer",
    "form": "input form",
    "h1": "heading level 1",
    "h2": "heading level 2",
    "h3": "heading level 3",
    "h4": "heading level 4",
    "h5": "heading level 5",
    "h6": "heading level 6",
    "head": "page head",
    "header": "page header",
    "hr": "horizontal rule",
    "html": "HTML root element",
    "i": "italic text",
    "iframe": "inline frame",
    "img": "image",
    "input": "form input",
    "ins": "inserted text",
    "kbd": "keyboard input",
    "label": "form label",
    "legend": "form fieldset caption",
    "li": "list item",
    "link": "external resource link",
    "main": "main content",
    "map": "image map",
    "mark": "highlighted text",
    "meta": "metadata",
    "meter": "measurement display",
    "nav": "navigation links",
    "noscript": "no-script fallback",
    "object": "embedded object",
    "ol": "ordered list",
    "optgroup": "option group",
    "option": "dropdown option",
    "output": "output result",
    "p": "paragraph",
    "picture": "responsive image container",
    "pre": "preformatted text",
    "progress": "progress indicator",
    "q": "inline quotation",
    "rp": "ruby text fallback",
    "rt": "ruby text annotation",
    "ruby": "ruby annotation",
    "s": "strikethrough text",
    "samp": "sample output",
    "script": "script code",
    "section": "section of content",
    "select": "dropdown selector",
    "small": "small text",
    "source": "media source",
    "span": "inline container",
    "strong": "strong emphasis",
    "style": "CSS style block",
    "sub": "subscript text",
    "summary": "details summary",
    "sup": "superscript text",
    "svg": "SVG graphic",
    "table": "data table",
    "tbody": "table body",
    "td": "table cell",
    "template": "HTML template",
    "textarea": "multiline text input",
    "tfoot": "table footer",
    "th": "table header cell",
    "thead": "table header",
    "time": "time or date",
    "title": "page title",
    "tr": "table row",
    "track": "media text track",
    "u": "underlined text",
    "ul": "unordered list",
    "var": "variable name",
    "video": "video player",
    "wbr": "word break opportunity"
}


def normalize_text_from_node(html_node):
    # Get all the text content from the node
    text = html_node.get_text(separator='\n', strip=True)

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Replace other whitespace (tabs, multiple spaces, etc.) with a single space
    # but preserve newlines
    text = re.sub(r'[^\S\r\n]+', ' ', text)
    # Trim whitespace from each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    # Trim the overall result
    return text.strip()


def tag_to_xpath(tag: Tag) -> str:
    """
    Given a BeautifulSoup Tag object, return its XPath.
    """
    elements = []
    current = tag

    while current is not None and isinstance(current, Tag):
        parent = current.parent
        if parent is None or not isinstance(parent, Tag):
            break

        # Find this tag's index among siblings of the same name
        siblings = [sib for sib in parent.find_all(current.name, recursive=False)]
        if len(siblings) == 1:
            index = ''
        else:
            index = f'[{siblings.index(current) + 1}]'

        elements.append(f'{current.name}{index}')
        current = parent

    # Reverse to get from root to target
    elements.reverse()
    return '/' + '/'.join(elements)


if __name__ == '__main__':
    # Example usage
    html = """
    <html>
      <body>
        <div>
          <p>Hello</p>
          <p id="target">World</p>
        </div>
      </body>
    </html>
    """

    soup = BeautifulSoup(html, 'html.parser')
    target_tag = soup.find('p', {'id': 'target'})
    print(tag_to_xpath(target_tag))
    # Output: /html/body/div/p[2]
