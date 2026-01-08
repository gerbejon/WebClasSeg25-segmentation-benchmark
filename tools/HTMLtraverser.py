import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# from docling import html_to_markdown

def get_dom_path(node, body):    # Calculate the DOM path for the node
    path_parts = []
    for parent in node.parents:
        if parent.name is None or parent == body.parent:
            break
        siblings = [sib for sib in parent.find_all(node.name, recursive=False)]
        if len(siblings) > 1:
            idx = siblings.index(node) + 1
            path_parts.append(f"{node.name}{idx}")
        else:
            path_parts.append(node.name)
        node = parent
    dom_path = '/'.join(reversed(path_parts))
    return dom_path, len(path_parts) -2

def get_max_depth(node, current_depth=0):
    if not hasattr(node, 'children'):
        return current_depth
    child_depths = [
        get_max_depth(child, current_depth + 1)
        for child in node.children
        if getattr(child, 'name', None) is not None
    ]
    return max(child_depths, default=current_depth)


def open_selenium_mhtml(page_id):
    # Open the MHTML file using Selenium
    
    chrome_options = Options()
    
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    print(f"Using ChromeDriver at: {driver.service.path}")

    
    # Open the MHTML file
    driver.get("http://160.85.252.105:59010/" + page_id + ".mhtml")
    time.sleep(2)  # Wait for the page to load

    # Resize window to full page height
    driver.set_window_size(width=2560, height=1440) 
    time.sleep(1)
    
    return driver

def check_if_visible(hyu_node, driver):
    """
    Check if the node with the given hyu is visible in the browser.
    """
    try:
        # Find the element by hyu attribute
        element = driver.find_element("xpath", f"//*[@hyu='{hyu_node}']")
        return element.is_displayed()
    
    except Exception as e:
        print(f"Error checking visibility for hyu {hyu_node}: {e}")
        return False
    
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


class HTMLTraverser:
    """
    A more flexible HTML traverser that supports breaking and jumping to siblings.
    """
    def __init__(self, root_node, page_id):
        self.root = root_node
        self.stack = [(root_node, [])]
        self.current_node = None
        self.current_parents = None
        self.selenium_driver = open_selenium_mhtml(page_id)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        
        # Pop the current node and its parents
        self.current_node, self.current_parents = self.stack.pop()
        
        # Add children to stack (in reverse order for left-to-right traversal)
        if hasattr(self.current_node, 'children'):
            children = [child for child in self.current_node.children 
                       if getattr(child, 'name', None) is not None]
            for child in reversed(children):
                self.stack.append((child, self.current_parents + [self.current_node]))
        
        # If the current node is not visible, skip it
        hyu = self.current_node.get('hyu', None)
        if hyu is None or not check_if_visible(hyu, self.selenium_driver):
            return self.__next__()
    
        # Simple path based on parent stack
        dom_path = self._get_simple_path()
        depth = len(self.current_parents)
        
        return self.current_node, self.current_parents, dom_path, depth
    
    def _get_simple_path(self):
        """
        Create a simple DOM path based on the parent stack.
        This doesn't handle sibling indexing like get_dom_path does.
        """
        if not self.current_node.name:
            return ""
        
        path_parts = []
        
        # Add current node
        # path_parts.append(self.current_node.name)
        
        # Add parents in reverse order
        for parent in reversed(self.current_parents):
            if parent.name:
                path_parts.insert(0, parent.name)
        
        return '/'.join(path_parts)
        
    def skip_subtree(self):
        """
        Skip the entire subtree (all descendants) of the current node.
        """
        if not self.current_node:
            return
        
        # Remove all nodes that are descendants of current node
        items_to_remove = []
        for item in self.stack:
            node, parents = item
            # If current_node is in the ancestry of this node, remove it
            if self.current_node in parents:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            if item in self.stack:
                self.stack.remove(item)
                
                
class HTMLNodeFinder:
    """
    Opens an HTML file and finds the node with a specific hyu attribute.
    """
    def __init__(self, html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            self.soup = BeautifulSoup(f, "html.parser")

    def find_by_hyu(self, hyu_value):
        """
        Returns the node with hyu=integer (as string or int) and its DOM path.
        """
        node = self.soup.find(attrs={"hyu": str(hyu_value)})
        if node:
            body = self.soup.body
            dom_path, _ = get_dom_path(node, body)
            return node, dom_path
        return None, None