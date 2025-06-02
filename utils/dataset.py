import torch
from torch.utils.data import Dataset
import numpy as np

class ReconstructableTextDataset(Dataset):

    def __init__(self, raw_texts: list[str], tokenizer, max_length, full_lists: list[list[str]], category_indices_list: list[list[int]], **identifiers):
        """
        Creates a dataset object that contains the usual input_ids and attention_mask, but also returns a B-length list of the original tokens 
         in the same position as the input ids, as well as any optional identifiers. Returning the original tokens is important for BPE 
         tokenizers as otherwise it's difficult to reconstruct the correct string later!

        Params:
            @raw_texts: A list of samples of text dataset.
            @tokenizer: A HF tokenizer object.
            @full_lists: A list of samples, each a sublist containing the elements.
            @category_indices_list: A list of samples, each a sublist containing the integers of full_list which contain the category-related elements.
            @ident_lists: Named lists such as q_indices = [...], sources = [...], each the same length as raw_texts. Will be identifiers. 
             These should contain useful identifiers that will be returned in the dataloader.

        Example:
            dl = DataLoader(
                ReconstructableTextDataset(['a', 'hello'], tokenizer, max_length = 768, q_indices = [0, 1]),
                batch_size = 2,
                shuffle = False,
                collate_fn = collate_fn
            )
        """
        tokenized = tokenizer(
            raw_texts,
            add_special_tokens = False,
            max_length = max_length,
            padding = 'max_length',
            truncation = True,
            return_offsets_mapping = True,
            return_tensors = 'pt'
        )

        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']
        self.offset_mapping = tokenized['offset_mapping']
        self.original_tokens = self._get_original_tokens(raw_texts)

        self.running_counts = [] # Int, cumulative #category so far
        self.is_category_tok = [] # Bool, token is a category word?
        self.list_tok_mask = [] # Bool, token is ANY list word (a superset of category words) - so we only store hidden states in the list, to reduce memory blowup
        self.word_start_tok_mask = [] # Bool, first sub-token of a word
        self.answer_pos = [] # Int, index of '(' token

        # ----------  Per-sample labels ----------
        for sample_idx, (
            text,
            offsets,
            words,
            cat_idxs,
        ) in enumerate(
            zip(raw_texts, self.offset_mapping, full_lists, category_indices_list)
        ):
            offsets_np = offsets.tolist()
            N = len(offsets_np)

            # --- 0) answer token ('(') ---
            ans_char = text.find("(")
            if ans_char == -1:
                raise ValueError(f"No '(' found in prompt of sample {sample_idx}.")
            try:
                # look for token whose span covers '(' character)
                ans_tok = next(
                    t for t, (s, e) in enumerate(offsets_np)     # offsets as list of tuples
                    if s <= ans_char < e                         # ← inclusive test
                )
            except StopIteration:
                raise ValueError(f"'(' appears to have been truncated (sample {sample_idx}). Increase max_length or drop this sample.")
            if ans_tok >= max_length:
                raise ValueError(f"Answer token beyond max_length (sample {sample_idx}).")
            self.answer_pos.append(ans_tok)

            # --- 1) deterministic character spans for each word ---
            word_spans = []
            cursor = text.find("[") + 2  # skip "[ "
            for w, word in enumerate(words):
                start = cursor
                end = start + len(word)
                word_spans.append((start, end))
                cursor = end + (1 if w < len(words) - 1 else 0)  # skip trailing space

            # --- 2) token → word mapping (-1 if not part of a list word) ---
            word_of_tok = [-1] * N
            for t, (s, e) in enumerate(offsets_np):
                if s == 0 and e == 0:  # PAD
                    break
                for w, (a, b) in enumerate(word_spans):
                    # overlap test handles leading-space subtokens
                    if s < b and e > a:
                        word_of_tok[t] = w
                        break

            # --- 3) cumulative running counts & masks ---
            cat_hit      = [int(w in cat_idxs) for w in range(len(words))]
            cum_by_word  = np.cumsum(cat_hit)

            run_cnt = []
            current = 0                       # ← last seen running total
            for w in word_of_tok:
                if w == -1:                   # token outside the list
                    run_cnt.append(current)   # keep previous total
                else:
                    current = int(cum_by_word[w])
                    run_cnt.append(current)

            is_cat  = [bool(w in cat_idxs) if w != -1 else False for w in word_of_tok]
            list_tok = [w != -1 for w in word_of_tok]

            # --- 4) word-start token mask ---
            wstart = torch.zeros(N, dtype=torch.bool)
            for w, (a, b) in enumerate(word_spans):
                try:
                    first_tok = next(
                        t for t, (s, e) in enumerate(offsets_np)
                        if s <= a < e                 # a = word start char
                    )                
                except StopIteration:
                    raise ValueError(f"Could not find any token that overlaps word '{words[w]}' in sample {sample_idx}."
                    )
                wstart[first_tok] = True

            # --- 5) sanity assertions ---
            rc_tensor = torch.tensor(run_cnt)
            if not torch.all(rc_tensor[1:] >= rc_tensor[:-1]):
                raise ValueError(f"running_count not monotone in sample {sample_idx}.")
            if rc_tensor.max().item() != len(cat_idxs):
                raise ValueError(f"Final running_count ({rc_tensor.max()}) ≠ true #category words ({len(cat_idxs)}) in sample {sample_idx}.")
            if not all(
                torch.tensor(is_cat, dtype=torch.bool)
                <= torch.tensor(list_tok, dtype=torch.bool)
            ):
                raise ValueError(f"is_category_tok not subset of list_tok_mask (sample {sample_idx}).")
            if wstart.sum().item() != len(words):
                raise ValueError(f"word_start_tok_mask has {wstart.sum().item()} starts but list has {len(words)} words (sample {sample_idx}).")

            # --- 6) pack tensors ---
            self.running_counts.append(torch.tensor(run_cnt, dtype=torch.long))
            self.is_category_tok.append(torch.tensor(is_cat, dtype=torch.bool))
            self.list_tok_mask.append(torch.tensor(list_tok, dtype=torch.bool))
            self.word_start_tok_mask.append(wstart)

        # stack per-sample lists into tensors
        self.running_counts = torch.stack(self.running_counts)
        self.is_category_tok = torch.stack(self.is_category_tok)
        self.list_tok_mask = torch.stack(self.list_tok_mask)
        self.word_start_tok_mask = torch.stack(self.word_start_tok_mask)
        self.answer_pos = torch.tensor(self.answer_pos, dtype=torch.long)

        # ----------  Original helper to store identifiers ----------
        self._ident_lists = identifiers  # Keep as dict for iteration
        n = len(raw_texts)
        for k, v in identifiers.items():
            if len(v) != n:
                raise ValueError(f"Length mismatch for '{k}': {len(v)} ≠ {n}")
            setattr(self, k, v) # Sets identifiers as keys.
    

    def _get_original_tokens(self, texts):
        """
        Return the original tokens associated with each B x N position. This is important for reconstructing the original text when BPE tokenizers are used. They 
         are returned in form [[seq1tok1, seq1tok2, ...], [seq2tok1, seq2tok2, ...], ...].
        
        Params:
            @input_ids: A B x N tensor of input ids.
            @offset_mapping: A B x N x 2 tensor of offset mappings. Get from `tokenizer(..., return_offsets_mapping = True)`.

        Returns:
            A list of length B, each with length N, containing the corresponding original tokens corresponding to the token ID at the same position of input_ids.
        """
        all_token_substrings = []
        for i in range(0, self.input_ids.shape[0]):
            token_substrings = []
            for j in range(self.input_ids.shape[1]): 
                start_char, end_char = self.offset_mapping[i][j].tolist()
                if start_char == 0 and end_char == 0: # When pads, offset_mapping might be [0, 0], so let's store an empty string for those positions.
                    token_substrings.append("")
                else:
                    original_substring = texts[i][start_char:end_char]
                    token_substrings.append(original_substring)
            
            all_token_substrings.append(token_substrings)

        return all_token_substrings

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'original_tokens': self.original_tokens[idx],
            "running_count": self.running_counts[idx],
            "is_category_tok": self.is_category_tok[idx],
            "list_tok_mask": self.list_tok_mask[idx],
            "word_start_tok_mask": self.word_start_tok_mask[idx],
            "answer_pos": self.answer_pos[idx],
        }
        for k, v in self._ident_lists.items(): # Attach metadata
            item[k] = v[idx]

        return item
    
def stack_collate(batch):
    """
    Custom collate function; returns everything in a dataset as a list except tensors, which are stacked. 
    """
    stacked = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        stacked[k] = torch.stack(vals, dim = 0) if torch.is_tensor(vals[0]) else vals
        
    return stacked