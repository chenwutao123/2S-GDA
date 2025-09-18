import pdb

import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import torchvision.transforms as T

try:
    from eda import eda, get_synonyms
except ImportError:
    print("Warning: eda.py not found. EDA and synonym features will be disabled.")
    def eda(text, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug):
        return [text]
    def get_synonyms(word):
        return []


class SGAttacker:
    def __init__(self, model, img_attacker, txt_attacker, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.05,
                 num_aug=4, use_eda_text_aug=False, use_sga_last_step=True):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd
        self.num_aug = num_aug
        self.use_eda_text_aug = use_eda_text_aug
        self.use_sga_last_step = use_sga_last_step

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        with torch.no_grad():
            origin_img_output = self.model.inference_image(
                self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]

        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_feats=img_supervisions, device=device)

        if self.use_eda_text_aug:
            with torch.no_grad():
                aug_adv_txts = []
                extended_txt2img = []

                for i, (txt, adv_txt) in enumerate(zip(txts, adv_txts)):
                    current_img_idx = txt2img[i]

                    eda_adv_txts = eda(adv_txt, self.alpha_sr, self.alpha_ri, self.alpha_rs, self.alpha_rd, self.num_aug)
                    aug_adv_txts.extend(eda_adv_txts)
                    extended_txt2img.extend([current_img_idx] * len(eda_adv_txts))

                    eda_orig_txts = eda(txt, self.alpha_sr, self.alpha_ri, self.alpha_rs, self.alpha_rd, self.num_aug)
                    aug_adv_txts.extend(eda_orig_txts)
                    extended_txt2img.extend([current_img_idx] * len(eda_orig_txts))

                adv_txts_input = self.txt_attacker.tokenizer(aug_adv_txts, padding='max_length', truncation=True,
                                                             max_length=max_length, return_tensors="pt").to(device)
                txts_output = self.model.inference_text(adv_txts_input)
                txt_supervisions_feat = txts_output['text_feat']
                txt_supervisions_embed = txts_output['text_embed']

        else:
            adv_txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                         max_length=max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                txts_output = self.model.inference_text(adv_txts_input)
                txt_supervisions_feat = txts_output['text_feat']
                txt_supervisions_embed = txts_output['text_embed']
            extended_txt2img = txt2img

        adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, extended_txt2img, device,
                                                       scales=scales, txt_feats=txt_supervisions_feat,
                                                       txt_embeds=txt_supervisions_embed,
                                                       txt_inputs=adv_txts_input)

        if self.use_sga_last_step:
            with torch.no_grad():
                adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
                img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_feats=img_supervisions, device=device)

        return adv_imgs, adv_txts


class ImageAttacker:
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, use_bsr=False, num_scale=5, num_block=3):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.use_bsr = use_bsr
        self.num_scale = num_scale
        self.num_block = num_block

    def loss_func_ITC(self, adv_imgs_embeds, txts_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss

    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None, txt_feats=None,
                          txt_embeds=None, txt_inputs=None):

        model.eval()

        b, _, _, _ = imgs.shape

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        for i in range(self.steps):
            adv_imgs.requires_grad_()
            scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device)

            effective_scales_num = scaled_imgs.shape[0] // b

            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.inference_image(scaled_imgs)

            adv_imgs_feats = adv_imgs_output['image_feat']
            adv_imgs_embeds = adv_imgs_output['image_embed']

            model.zero_grad()
            with torch.enable_grad():
                loss_ITC = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i_scale in range(effective_scales_num):
                    loss_item = self.loss_func_ITC(adv_imgs_feats[i_scale * b:i_scale * b + b], txt_feats, txt2img)
                    loss_ITC += loss_item

            loss_ITC.backward()

            grad = adv_imgs.grad
            grad_total = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

            perturbation = self.step_size * grad_total.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps),
                                 imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        return adv_imgs

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        result = []
        if scales is not None or not self.use_bsr:
            result.append(imgs)

        if scales is not None:
            ori_shape = (imgs.shape[-2], imgs.shape[-1])
            reverse_transform = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC,
                                                  antialias=True)
            for ratio in scales:
                scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
                scale_transform = transforms.Resize(scale_shape, interpolation=transforms.InterpolationMode.BICUBIC,
                                                    antialias=True)
                scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
                scaled_imgs = scale_transform(scaled_imgs)
                scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
                reversed_imgs = reverse_transform(scaled_imgs)
                result.append(reversed_imgs)

        if self.use_bsr:
            for _ in range(self.num_scale):
                bsr_img = self.shuffle(imgs.clone())
                result.append(bsr_img)

        return torch.cat(result, 0) if len(result) > 0 else imgs


    def get_length(self, length):
        rand = np.random.uniform(2, size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips

    def image_rotation(self, x):
        rotation_transform = T.RandomRotation(degrees=(-24, 24),
                                              interpolation=T.InterpolationMode.BILINEAR)
        return rotation_transform(x)

    def shuffle(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])

        processed_strips = []
        for x_strip in x_strips:
            rotated = self.image_rotation(x_strip)
            shuffled = self.shuffle_single_dim(rotated, dims[1])
            concatenated = torch.cat(shuffled, dim=dims[1])
            processed_strips.append(concatenated)

        return torch.cat(processed_strips, dim=dims[0])


filter_words = {'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·'}


class TextAttacker:
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, k_perturbation_candidates=3,
                 use_synonyms=False, use_non_greedy_replacement=False, device='cpu'):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.k_perturbation_candidates = k_perturbation_candidates

        self.use_synonyms = use_synonyms

        self.use_non_greedy_replacement = use_non_greedy_replacement


    def img_guided_attack(self, net, texts, img_feats=None, device='cpu'):
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []

        print("Using Word Substitution Text Attack.")
        mlm_logits = self.ref_net(text_inputs.input_ids,
                                  attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        for i, text in enumerate(texts):
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            original_words_copy = copy.deepcopy(words)

            print('Importance Fusion Method', [words[top_index[0]] for top_index in list_of_index])

            if self.use_non_greedy_replacement:
                potential_replacements = []
                words_processed_for_k = 0

                for top_index_tuple in list_of_index:
                    original_word_idx = top_index_tuple[0]
                    tgt_word = words[original_word_idx]

                    if tgt_word in filter_words:
                        continue
                    if keys[original_word_idx][0] > self.max_length - 2:
                        continue

                    if words_processed_for_k >= self.k_perturbation_candidates:
                        break

                    mlm_substitutes_ids = word_predictions[i, keys[original_word_idx][0]:keys[original_word_idx][1]]
                    mlm_substitutes_scores = word_pred_scores_all[i, keys[original_word_idx][0]:keys[original_word_idx][1]]
                    mlm_candidates = get_substitues(mlm_substitutes_ids, self.tokenizer, self.ref_net, 1,
                                                    mlm_substitutes_scores, self.threshold_pred_score)

                    synonym_candidates = []
                    if self.use_synonyms:
                        synonym_candidates = get_synonyms(tgt_word)

                    all_potential_substitutes = set()
                    for cand in mlm_candidates:
                        if '##' not in cand and cand not in filter_words and cand != tgt_word:
                            all_potential_substitutes.add(cand)
                    if self.use_synonyms:
                        for cand in synonym_candidates:
                            if '##' not in cand and cand not in filter_words and cand != tgt_word:
                                all_potential_substitutes.add(cand)

                    substitutes_for_eval = list(all_potential_substitutes)

                    if not substitutes_for_eval:
                        print(f"No valid substitutes found for '{tgt_word}' after filtering. Skipping.")
                        continue

                    replace_texts_for_eval = []
                    available_substitutes_for_eval = []

                    replace_texts_for_eval.append(' '.join(original_words_copy))
                    available_substitutes_for_eval.append(tgt_word)

                    print(f'{tgt_word} substitutes (MLM{" + Synonyms" if self.use_synonyms else ""})', end=': ')
                    for sub_word in substitutes_for_eval:
                        print(sub_word, end=',')
                        temp_replace = copy.deepcopy(original_words_copy)
                        temp_replace[original_word_idx] = sub_word
                        replace_texts_for_eval.append(' '.join(temp_replace))
                        available_substitutes_for_eval.append(sub_word)
                    print(end='\n')

                    replace_text_input = self.tokenizer(replace_texts_for_eval, padding='max_length', truncation=True,
                                                        max_length=self.max_length, return_tensors='pt').to(device)

                    replace_output = net.inference_text(replace_text_input)
                    if self.cls:
                        replace_embeds = replace_output['text_feat'][:, 0, :]
                    else:
                        replace_embeds = replace_output['text_feat'].flatten(1)

                    loss_candidates = self.loss_func(replace_embeds, img_feats, i)
                    candidate_idx = loss_candidates.argmax()

                    best_loss_for_this_word = loss_candidates[candidate_idx].item()
                    chosen_substitute_for_this_word = available_substitutes_for_eval[candidate_idx]

                    if chosen_substitute_for_this_word != tgt_word:
                        potential_replacements.append({
                            'original_word_idx': original_word_idx,
                            'original_word': tgt_word,
                            'chosen_substitute': chosen_substitute_for_this_word,
                            'max_loss': best_loss_for_this_word
                        })
                    else:
                        print(f"Best substitute for '{tgt_word}' was original word. Not considering for top K selection.")

                    words_processed_for_k += 1

                final_words_for_current_text = copy.deepcopy(words)
                if potential_replacements:
                    best_overall_replacement = max(potential_replacements, key=lambda x: x['max_loss'])

                    original_word_idx_to_replace = best_overall_replacement['original_word_idx']
                    chosen_word_to_apply = best_overall_replacement['chosen_substitute']
                    original_word_to_replace = best_overall_replacement['original_word']

                    final_words_for_current_text[original_word_idx_to_replace] = chosen_word_to_apply

                    print(f'Overall best replacement for text {i}: target "{original_word_to_replace}" (at index {original_word_idx_to_replace}) replaced with "{chosen_word_to_apply}" with loss {best_overall_replacement["max_loss"]:.4f}\n')
                else:
                    print(f"No valid replacement found among top {self.k_perturbation_candidates} important words for text {i}. Keeping original text.\n")

                final_adverse.append(' '.join(final_words_for_current_text))

            else:
                final_words = copy.deepcopy(words)
                change = 0

                for top_index_tuple in list_of_index:
                    if change >= self.num_perturbation:
                        break

                    original_word_idx = top_index_tuple[0]
                    tgt_word = words[original_word_idx]

                    if tgt_word in filter_words:
                        continue
                    if keys[original_word_idx][0] > self.max_length - 2:
                        continue

                    mlm_substitutes_ids = word_predictions[i, keys[original_word_idx][0]:keys[original_word_idx][1]]
                    mlm_substitutes_scores = word_pred_scores_all[i, keys[original_word_idx][0]:keys[original_word_idx][1]]
                    mlm_candidates = get_substitues(mlm_substitutes_ids, self.tokenizer, self.ref_net, 1,
                                                    mlm_substitutes_scores, self.threshold_pred_score)

                    synonym_candidates = []
                    if self.use_synonyms:
                        synonym_candidates = get_synonyms(tgt_word)

                    all_potential_substitutes = set()
                    for cand in mlm_candidates:
                        if '##' not in cand and cand not in filter_words and cand != tgt_word:
                            all_potential_substitutes.add(cand)
                    if self.use_synonyms:
                        for cand in synonym_candidates:
                            if '##' not in cand and cand not in filter_words and cand != tgt_word:
                                all_potential_substitutes.add(cand)

                    substitutes_for_eval = list(all_potential_substitutes)

                    if not substitutes_for_eval:
                        print(f"No valid substitutes found for '{tgt_word}' after filtering. Skipping.")
                        continue

                    replace_texts_for_eval = []
                    available_substitutes_for_eval = []

                    replace_texts_for_eval.append(' '.join(final_words))
                    available_substitutes_for_eval.append(tgt_word)

                    print(f'{tgt_word} substitutes (MLM {"+ Synonyms" if self.use_synonyms else ""})', end=': ')
                    for sub_word in substitutes_for_eval:
                        print(sub_word, end=',')
                        temp_replace = copy.deepcopy(final_words)
                        temp_replace[original_word_idx] = sub_word
                        replace_texts_for_eval.append(' '.join(temp_replace))
                        available_substitutes_for_eval.append(sub_word)
                    print(end='\n')

                    replace_text_input = self.tokenizer(replace_texts_for_eval, padding='max_length', truncation=True,
                                                        max_length=self.max_length, return_tensors='pt').to(device)

                    replace_output = net.inference_text(replace_text_input)
                    if self.cls:
                        replace_embeds = replace_output['text_feat'][:, 0, :]
                    else:
                        replace_embeds = replace_output['text_feat'].flatten(1)

                    loss_candidates = self.loss_func(replace_embeds, img_feats, i)
                    candidate_idx = loss_candidates.argmax()

                    chosen_substitute = available_substitutes_for_eval[candidate_idx]
                    final_words[original_word_idx] = chosen_substitute

                    if chosen_substitute != tgt_word:
                        change += 1
                        print(f'target: "{tgt_word}" replaced with "{chosen_substitute}" (Loss: {loss_candidates[candidate_idx].item():.4f})')
                    else:
                        print(f"Best substitute for '{tgt_word}' was original word. No change made.")

                final_adverse.append(' '.join(final_words))
                print(f"Final adversarial text for text {i} (Greedy): '{' '.join(final_words)}'\n")

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def _tokenize(self, text):
        words = text.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    c_loss = nn.CrossEntropyLoss(reduction='none')
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

