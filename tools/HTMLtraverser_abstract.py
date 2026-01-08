import abc
from bs4 import BeautifulSoup

class BaseHTMLTraverser(abc.ABC):
    def __init__(self, root_node):
        self.root = root_node
        self.stack = [(root_node, [])]
        self.current_node = None
        self.current_parents = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stack:
            raise StopIteration
        self.current_node, self.current_parents = self.stack.pop()

        children = [c for c in getattr(self.current_node, "children", [])
                    if getattr(c, "name", None) is not None]
        for child in reversed(children):
            self.stack.append((child, self.current_parents + [self.current_node]))

        if not self.is_visible(self.current_node):
            return self.__next__()

        return self.current_node, self.current_parents, self._get_simple_path(), len(self.current_parents)

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


    @abc.abstractmethod
    def is_visible(self, node):
        """Determine if the node is visible."""
        pass

    def _get_simple_path(self):
        return "/".join(p.name for p in self.current_parents if p.name)
