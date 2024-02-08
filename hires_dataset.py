import torch
from torch_geometric.data import Data, InMemoryDataset


class HiRESDataset(InMemoryDataset):
    """Dataset for combined Hi-C and RNA-seq data, indexed by cell."""
    def __init__(self, root, gene_indices=None):
        self.gene_indices = gene_indices
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        print("processed paths: ", type(self.processed_paths[0]))
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["raw_data.pt"]

    def download(self):
        raise NotImplementedError("Please ensure that root/raw/raw_data.pt exists.")

    @property
    def processed_file_names(self):
        return ["dataset.pt"]

    def process(self):
        """Process raw data into list of Data objects."""
        raw_data = torch.load(Path(self.raw_dir) / "raw_data.pt")
        data_list = self.process_data(raw_data, self.gene_indices)
        self.save(data_list, self.processed_paths[0])
    
    # end region

    # region Helper functions
    def process_data(self, raw_data, gene_indices):
        """Construct list of Data objects from intermediate data.

        Each Data object consists of a graph constructed from the adjacency matrix from
        the HIC data. Each edge has a single attribute, the edge weight. The labels are
        the genes of interest selected from the RNASEQ vector.

        Args:
            raw_data (List[Dict]): list of dictionaries of data loaded by load_raw_data.
            gene_indices (List): indices of genes in HiRES vector to use for prediction.
        
        Returns:
            List: a list of Data objects.
        """
        def dict_to_data(item):
            hic, rnaseq = item["hic"], item["rnaseq"]
            edges = hic.nonzero().t().contiguous()  # 2xN tensor where each column represents an edge
            edge_weights = hic[hic.nonzero(as_tuple=True)]  # Values from adjacency matrix

            return Data(
                num_nodes=hic.shape[0],
                edge_index=edges,
                edge_attr=edge_weights,
                y=rnaseq[gene_indices]
            )

        # Convert each dictionary in the intermediate data to a Data object
        return list(map(dict_to_data, raw_data))
