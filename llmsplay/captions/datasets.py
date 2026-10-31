"""
PyTorch dataset for image captioning on Flickr-style data.

Data flow:
    image paths + tokenized captions
        -> one (image_path, caption_tokens) pair per caption
        -> __getitem__: load/crop/resize/normalize image, wrap caption with
           <start>/<end>, pad, map tokens to vocabulary ids
        -> (image_tensor, input_ids, target_ids, attention_mask)

The target sequence is the input shifted one position left, so the model
learns to predict the next token (teacher forcing).
"""

import json, torch
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FlickrD(Dataset):
    """
    Image-caption dataset producing one sample per (image, caption) pair.
    """

    def __init__(
        self,
        images: List[str],
        captions: List[List[List[str]]],
        word2idx: Dict[str, int],
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        images : list of str
            Image file paths; 'images[i]' is described by 'captions[i]'.
        captions : list of list of list of str
            For each image, a list of captions, each one a list of word tokens.
        word2idx : dict of str to int
            Vocabulary mapping a token to its integer id. Must contain
            '<pad>' (0), '<start>' (1), '<end>' (2) and '<unk>' (3).
        """
        
        self.images: List[str] = images
        self.captions: List[List[List[str]]] = captions
        self.word2idx: Dict[str, int] = word2idx
        # Fixed sequence length after padding
        self._max_len: int = 50
        # Images are resized to image_size x image_size
        self.image_size: int = 128
        self._image_transform: transforms.Compose = self._construct_image_transform(
            self.image_size
        )
        # Flat list of (image_path, caption_tokens) pairs
        self._data: List[Tuple[str, List[str]]] = self._create_input_label_mappings()
        self._dataset_size: int = len(self._data)
        # Special token ids and strings (ids must match word2idx)
        self._start_idx: int = 1
        self._end_idx: int = 2
        self._pad_idx: int = 0
        self._UNK_idx: int = 3
        self._START_token: str = "<start>"
        self._END_token: str = "<end>"
        self._PAD_token: str = "<pad>"
        self._UNK_token: str = "<unk>"

    def _construct_image_transform(self, image_size: int) -> transforms.Compose:
        """
        Build the PIL -> normalized tensor transform used for every image.

        Parameters
        ----------
        image_size : int
            Side length of the square images (not used by the transform itself).

        Returns
        -------
        transforms.Compose
            'ToTensor' followed by ImageNet 'Normalize'.
        """
        # ImageNet normalization statistics
        normalize: transforms.Normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        preprocessing: transforms.Compose = transforms.Compose(
            [
                transforms.ToTensor(),  # PIL [H, W, C] uint8 -> tensor [C, H, W] in [0, 1]
                normalize,
            ]
        )
        return preprocessing

    def _load_and_process_images(self) -> Dict[str, torch.Tensor]:
        """
        Eagerly load and preprocess every image.

        Not used by '__getitem__', which loads images lazily instead.

        Returns
        -------
        dict of str to torch.Tensor
            Mapping from image path to its preprocessed tensor '[3, H, W]'.
        """
        # Load images
        images_raw: List[Image.Image] = [
            center_crop(Image.open(path)).resize((self.image_size, self.image_size))
            for path in self.images
        ]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        image_tensors: List[torch.Tensor] = [
            self._image_transform(img) for img in images_raw
        ]
        images_processed: Dict[str, torch.Tensor] = {
            img_name: img_tensor
            for img_name, img_tensor in zip(self.images, image_tensors)
        }
        return images_processed

    def _group_captions(self) -> Dict[str, List[List[str]]]:
        """
        Map each image path to its list of tokenized captions.

        Returns
        -------
        dict of str to list of list of str
            Image path -> captions, each caption a list of tokens.
        """
        grouped_captions: Dict[str, List[List[str]]] = {
            self.images[i]: self.captions[i] for i in range(len(self.images))
        }
        return grouped_captions

    def _create_input_label_mappings(self) -> List[Tuple[str, List[str]]]:
        """
        Flatten captions so each caption becomes its own sample.

        Returns
        -------
        list of tuple of (str, list of str)
            '(image_path, caption_tokens)' pairs; an image with 5 captions
            yields 5 pairs.
        """
        processed_data: List[Tuple[str, List[str]]] = []
        img: str
        caps: List[List[str]]
        for img, caps in self._group_captions().items():
            cap: List[str]
            for cap in caps:
                pair: Tuple[str, List[str]] = (img, cap)
                processed_data.append(pair)
        return processed_data

    def _load_and_prepare_image(self, image_name: str) -> torch.Tensor:
        """
        Load one image, center-crop it to a square, resize and normalize it.

        Parameters
        ----------
        image_name : str
            Path to the image file.

        Returns
        -------
        torch.Tensor
            Normalized image of shape '[3, image_size, image_size]'.
        """
        img_pil: Image.Image = center_crop(Image.open(image_name)).resize(
            (self.image_size, self.image_size)
        )
        image_tensor: torch.Tensor = self._image_transform(img_pil)
        return image_tensor

    def __len__(self) -> int:
        """
        Return the dataset size.

        Returns
        -------
        int
            Number of (image, caption) pairs.
        """
        return self._dataset_size

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build one training sample.

        Parameters
        ----------
        index : int
            Position of the (image, caption) pair in the dataset.

        Returns
        -------
        image_tensor : torch.Tensor
            Normalized image of shape '[3, image_size, image_size]'.
        input_tokens : torch.Tensor
            Token ids '<start> w1 ... wn' + padding, shape '[max_len]'.
        tgt_tokens : torch.Tensor
            Token ids 'w1 ... wn <end>' + padding, shape '[max_len]'.
        attn_mask : torch.Tensor
            Bool mask of shape '[max_len]'; True for real tokens, False for padding.
        """
        # Extract the caption data
        image_id: str
        tokens: List[str]
        image_id, tokens = self._data[index]
        # Load and preprocess image
        image_tensor: torch.Tensor = self._load_and_prepare_image(image_id)
        # Truncate, normalize and wrap the caption with special tokens
        tokens = tokens[: self._max_len]
        tokens = [token.strip().lower() for token in tokens]
        tokens = [self._START_token] + tokens + [self._END_token]
        # Shift by one: input predicts the next token in target
        input_words: List[str] = tokens[:-1].copy()
        tgt_words: List[str] = tokens[1:].copy()

        # Number of words in the input token
        sample_size: int = len(input_words)
        padding_size: int = self._max_len - sample_size

        # Right-pad both sequences to max_len
        if padding_size > 0:
            padding_vec: List[str] = [self._PAD_token for _ in range(padding_size)]
            input_words += padding_vec.copy()
            tgt_words += padding_vec.copy()

        # Map tokens to vocabulary ids; unknown words become <unk>
        input_ids: List[int] = [
            self.word2idx.get(token, self._UNK_idx) for token in input_words
        ]
        tgt_ids: List[int] = [
            self.word2idx.get(token, self._UNK_idx) for token in tgt_words
        ]

        input_tokens: torch.Tensor = torch.Tensor(input_ids).long()
        tgt_tokens: torch.Tensor = torch.Tensor(tgt_ids).long()

        # Padding mask: True on real tokens, False on padding
        attn_mask: torch.Tensor = torch.zeros(
            [
                self._max_len,
            ]
        )
        attn_mask[:sample_size] = 1.0
        attn_mask = attn_mask.bool()

        return image_tensor, input_tokens, tgt_tokens, attn_mask


def center_crop(img: Image.Image) -> Image.Image:
    """
    Crop the largest centered square from 'img'.
    """
    
    width: int
    height: int
    width, height = img.size
    new: int = min(width, height)
    left: float = (width - new) / 2
    top: float = (height - new) / 2
    right: float = (width + new) / 2
    bottom: float = (height + new) / 2
    im: Image.Image = img.crop((left, top, right, bottom))
    
    # Return the cropped image
    return im
