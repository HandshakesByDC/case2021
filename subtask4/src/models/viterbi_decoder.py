import torch

class ViterbiDecoder():
    def __init__(self, id2tag, pad_token_label_id, device):
        self.n_labels = len(id2tag)
        self.pad_token_label_id = pad_token_label_id
        self.label_map = id2tag

        self.transitions = torch.zeros([self.n_labels, self.n_labels], device=device) # pij: p(j -> i)
        for i in range(self.n_labels):
            for j in range(self.n_labels):
                if self.label_map[i][0] == "I" and self.label_map[j][-3:] != self.label_map[i][-3:]:
                    self.transitions[i, j] = -10000
                    # print(f"{self.label_map[i]} -> {self.label_map[j]} not allowed")

    def forward(self, logprobs, attention_mask, label_ids):
        active_tokens = (attention_mask == 1) & (label_ids != self.pad_token_label_id)

        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        # scores = []
        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][active_tokens[idx]] # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0] # n_labels

            for j in range(1, len(logprob_i)): # for tag_feat in feat:
                next_label_var = forward_var + self.transitions # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1) # n_labels

                logp_j = logprob_i[j] # n_labels
                forward_var = viterbivars_t + logp_j # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            # terminal_var = forward_var

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            # path_score = path_score.item()
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)
            # scores.append(path_score)

        return label_seqs #, scores
