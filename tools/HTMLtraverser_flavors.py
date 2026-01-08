import os.path
import time
from tools.HTMLtraverser_abstract import BaseHTMLTraverser
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class SeleniumHTMLTraverser(BaseHTMLTraverser):
    def __init__(self, mhtml_url):
        driver = self._open_selenium_mhtml(mhtml_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        self.driver = driver
        super().__init__(soup)

    def _open_selenium_mhtml(self, page_id):
        options = Options()
        options.add_argument("--disable-web-security")
        options.add_argument("--headless=new")
        options.add_experimental_option("detach", True)
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        print(f"Using ChromeDriver at: {driver.service.path}")
        driver.get("http://160.85.252.105:59010/" + page_id + ".mhtml")
        # driver.get('file://' +os.path.abspath("./" + page_id + ".mhtml"))
        time.sleep(2)  # Wait for the page to load

        # Resize window to full page height
        driver.set_window_size(width=2560, height=1440)
        time.sleep(1)
        return driver

    def is_visible(self, node):
        hyu = node.get("hyu")
        if not hyu:
            return False
        try:
            el = self.driver.find_element("xpath", f"//*[@hyu='{hyu}']")
            return el.is_displayed()
        except:
            return False



class StringHTMLTraverser(BaseHTMLTraverser):
    def __init__(self, html_string):
        soup = BeautifulSoup(html_string, "lxml")  # lxml is much faster
        super().__init__(soup)

    def is_visible(self, node):
        # Without a browser, define "visible" as: not style="display:none" and has text or children
        style = node.get("style", "")
        if "display:none" in style:
            return False
        return True