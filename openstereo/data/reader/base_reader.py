from torch.utils.data import Dataset


class BaseReader(Dataset):
    def __init__(self, root, list_file):
        self.root = root
        self.list_file = list_file
        self.data_list = self.load_anno()

    def load_anno(self):
        data_list = []
        with open(self.list_file, 'r') as fp:
            data_list.extend([x.strip().split(' ') for x in fp.readlines()])
        return data_list

    def item_loader(self, item):
        """
        Load a single item from the dataset.
        Args:
            item: list of str, [left_img_path, right_img_path, disp_img_path] or
            any other formate you defined in load_anno.

        Returns:
            sample: dict, {'left': left_img, 'right': right_img, 'disp': disp_img}
            or any other key-value your net needs and be sure that your transform
            function can handle.
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        item = self.data_list[idx]
        sample = self.item_loader(item)
        return sample

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Data root: {}\n'.format(self.root)
        repr_str += ' ' * 4 + 'Anno file: {}\n'.format(self.list_file)
        repr_str += ' ' * 4 + 'Data length: {}\n'.format(self.__len__())
        return repr_str
