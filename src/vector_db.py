import voyager as vg
from numpy.typing import NDArray


class VectorDatabase:
    def __init__(self):
        self.db_name = None
        self.index = None
        self.n_dim = None
        self.mode = None
        pass

    def create_new_index(self, n_dim: int):
        self.n_dim = n_dim
        self.index = vg.Index(vg.Space.Cosine, num_dimensions=self.n_dim)

    def save_index_on_disk(self, index_name: str):
        if self.index is None:
            print("Index is not initiated. Cannot save the index...")
        else:
            self.index.save(index_name)
            print(f"Index saved successfully at location: {index_name}")

    def load_index_from_disk(self, index_name: str, mode: str = "r"):
        self.mode = mode
        self.index = vg.Index.load(index_name)

    def add_array_to_index(self, embedding: NDArray) -> bool:
        if self.index is None:
            print("Index is not initiated. Cannot add to index..")
            success_flag = False
        elif self.mode == "r":
            print("Index is read only. Cannot add to index...")
            succes_flag = False
        else:
            _ = self.index.add_item(embedding)
            success_flag = True

        return success_flag
