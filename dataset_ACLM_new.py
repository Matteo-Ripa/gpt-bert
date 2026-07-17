import torch
import ast
import random
import pandas as pd
import ast

class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens, counts=None):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        if counts is not None:
            counts = counts.float()
            counts[tokens < self.n_special_tokens] = float('-inf')
            counts_p = torch.nn.functional.softmax(counts, dim=0)
            mask_ratios = mask_ratios * counts_p

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.indices = torch.randperm(n_segments)
        self.index = 0

    def get_random_index(self):
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0

        index = self.indices[self.index]
        self.index += 1

        return index


class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, seq_length, rank, world_size):
        self.path = input_file
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        #with open(input_file, "r", encoding="utf-8") as file:
        #    lines = file.readlines()
        #documents = torch.load(input_file)

        #documents = [torch.tensor(eval(line.split(": ")[1].strip()), dtype=torch.long) for line in lines if line.strip() ]
        if isinstance(input_file,pd.DataFrame):
            df = input_file

            self.segments = [torch.tensor(eval(row), dtype=torch.long) for row in df["Tokens"]]
            
        else:
            #input_file.endswith(".txt"):
            with open(input_file, "r", encoding="utf-8") as file:
                lines = file.readlines()

            self.segments = [torch.tensor(ast.literal_eval(line.strip()), dtype=torch.long)for line in lines if line.strip()]
        self.mask_counts = [
            torch.zeros_like(segment)
            for segment in self.segments
        ]
        self.random_index = RandomIndex(len(self.segments))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = tokens.size(0)
        #seq_length = min(self.seq_length, tokens.size(0))
        #tokens = tokens[:seq_length].long()
        #self.counts[index][:seq_length] += 1

        mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index])#[:seq_length])
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)
        #self.mask_counts[index][:seq_length][target_ids != -100] += 1
        self.mask_counts[index][target_ids != -100] += 1
        #input_ids = torch.cat([
        #    torch.LongTensor([self.cls_index]),
        #    input_ids
        #])
        #target_ids = torch.cat([
        #    torch.LongTensor([-100]),
        #    target_ids
        #])
        #attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
        pad_token_id = self.pad_index
        padding_mask = (tokens != pad_token_id)
        
        attention_mask = padding_mask.unsqueeze(0) & padding_mask.unsqueeze(1)
        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def set_global_step(self, global_step):
        self.global_step = global_step

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / self.args.max_steps
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def show_random_item(self, tokenizer):
        input_ids, target_ids, attention_mask, real_mask_p = self.__getitem__(torch.randint(0, len(self), []).item())
        ######################################
        print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
        for i in input_ids.tolist():
            token = tokenizer.id_to_token(i)
            if token is None:
                print(f"Invalid token ID: {i}", flush=True)

        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(str(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(tokenizer.id_to_token(i) if i != -100 else "-100" for i in target_ids.tolist()), flush=True)
        print()
        print(real_mask_p, flush=True)


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args, seq_length, rank, world_size):
        self.path = input_file
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        if isinstance(input_file,pd.DataFrame):
            df = input_file
            self.segments = [torch.tensor(eval(row), dtype=torch.long) for row in df["Tokens"]]


        elif input_file.endswith(".txt"):
            with open(input_file, "r", encoding="utf-8") as file:
                lines = file.readlines()

            self.segments = [torch.tensor(ast.literal_eval(line.strip()), dtype=torch.long)for line in lines if line.strip()]
        
        #if rank is not None:
        #    self.segments = self.segments[rank::world_size]
        self.counts = [
            torch.zeros_like(segment)
            for segment in self.segments
        ]
        self.random_index = RandomIndex(len(self.segments))
        
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        seq_length = tokens.size(0)
        #print('tokens size:', tokens.size(0))
        '''seq_length = min(self.seq_length, tokens.size(0))
        self.counts[index][:seq_length] += 1

        input_ids = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long()
        ])
        target_ids = torch.cat([
            torch.LongTensor([-100]),
            tokens[:seq_length].long()
        ])'''

        input_ids = tokens #[:-1]
        #print('input_ids size:', input_ids.size(0))
        target_ids = tokens #[1:]

        pad_index = self.pad_index

        padding_mask = (input_ids != pad_index)
        #attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        '''while self.seq_length - input_ids.size(0) > 1:
            index = self.random_index.get_random_index()
            tokens = self.segments[index].long()
            seq_length = min(self.seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                conv_weight = torch.ones(1, 1, seq_length)
                summed_counts = torch.nn.functional.conv1d(
                    self.counts[index].view(1, 1, -1).float(),
                    conv_weight
                ).squeeze()
                offset = torch.argmin(summed_counts)

            tokens = tokens[offset:offset + seq_length]
            self.counts[index][offset:offset+seq_length] += 1

            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.cls_index]),
                tokens
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100]),
                tokens
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
            )

        padding_length = self.seq_length - input_ids.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            ) '''

        # make the attention mask causal
        #attention_mask = attention_mask.tril()
        #attention_mask = ~attention_mask

        #input_ids = input_ids[:-1]
        #target_ids = target_ids[1:]
        #attention_mask = attention_mask[:-1, :-1]
        causal_mask = torch.tril(torch.ones(len(input_ids), len(input_ids), dtype=torch.bool))
        attention_mask = causal_mask & padding_mask.unsqueeze(0) & padding_mask.unsqueeze(1)
        attention_mask = ~attention_mask
        #print('attention_mask:', attention_mask.size())
        attention_mask = attention_mask[:-1, :-1]
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        #print('attention_mask:', attention_mask.size())
        return input_ids, target_ids, attention_mask, torch.zeros([])

    def set_global_step(self, global_step):
        self.global_step = global_step

    def show_random_item(self, tokenizer):
        input_ids, target_ids, attention_mask, real_mask_p = self.__getitem__(torch.randint(0, len(self), []).item())
        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(str(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(tokenizer.id_to_token(i) if i != -100 else "-100" for i in target_ids.tolist()), flush=True)
        print()
        print(real_mask_p, flush=True)


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, input_file: str, tokenizer, args):
        self.path = input_file
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens
        self.global_step = 0 ##added this line

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.sep_index = tokenizer.token_to_id("</s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

        #documents = torch.load(input_file)
        if isinstance(input_file,pd.DataFrame):
            df = input_file
            self.segments = [torch.tensor(eval(row), dtype=torch.long) for row in df["Tokens"]]

        elif input_file.endswith(".txt"):
            with open(input_file, "r", encoding="utf-8") as file:
                lines = file.readlines()

            self.segments = [torch.tensor(ast.literal_eval(line.strip()), dtype=torch.long)for line in lines if line.strip()]
            self.segments = self.segments[args.rank::args.world_size]
            random.seed(args.rank)
            #self.segments = [torch.tensor(eval(line.split(": ")[1].strip()), dtype=torch.long) for line in lines if line.strip()]
        else:
            print("This dataset only supports txt or dataframe as an input file")

        '''self.segments = [
            document[offset : offset + self.seq_length - 2]
            for document in documents
            for offset in range(0, len(document), self.seq_length - 2)
            if len(document) > 0 and len(document) - offset > 1
        ]'''
        #if input_file.endswith(".txt") and hasattr(args, "rank"):
        #    self.segments = self.segments[args.rank::args.world_size]
        #    random.seed(args.rank)
        #else:
        #    random.seed(args.seed)
        #random.shuffle(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        tokens = self.segments[index]
        #seq_length = min(self.seq_length - 2, tokens.size(0))

        '''segment = torch.cat([
            torch.LongTensor([self.cls_index]),
            tokens[:seq_length].long()
        ])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
        '''
        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)

        '''padding_length = self.seq_length - segment.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            attention_mask = torch.block_diag(
                attention_mask,
                torch.zeros(padding_length, padding_length, dtype=torch.bool)
            )
        '''
        pad_mask = (input_ids != self.pad_index)
        attention_mask = pad_mask.unsqueeze(0) & pad_mask.unsqueeze(1)

        attention_mask = ~attention_mask

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def set_global_step(self, global_step):
        self.global_step = global_step   ######added this function

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = 0.15
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + 0.5)), largest=False).values.max().item()

        mask = mask_ratios < mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p
