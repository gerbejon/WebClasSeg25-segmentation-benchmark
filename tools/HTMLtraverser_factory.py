from tools.HTMLtraverser_flavors import SeleniumHTMLTraverser, StringHTMLTraverser


class HTMLTraverserFactory:
    @staticmethod
    def create(source, mode="selenium"):
        if mode == "selenium":
            return SeleniumHTMLTraverser(source)
        elif mode == "string":
            return StringHTMLTraverser(source)
        else:
            raise ValueError(f"Unknown mode: {mode}")