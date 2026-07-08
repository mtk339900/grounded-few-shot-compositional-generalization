"""
Grounded Few-Shot English Learner — Multi-Stage Curriculum
==============================================================
Stage 1  : direct classification (one-hot categories) — validated baseline
Stage 2  : learned embeddings instead of fixed categories
Stage 3  : full sentences with no explicit concept labels (seq2seq + attention)
Stage 4  : paraphrase understanding (contrastive alignment)
Stage 5  : two-sentence paragraphs (two linked facts)
Stage 6  : single held-out compositional split
Stage 7  : GRU baseline (no attention, no factored encoding)
Stage 8  : multi-split compositional benchmark
Stage 9  : ablation study (isolates each architectural component)
Stage 10 : multi-seed evaluation harness
Stage 11 : Transformer baselines (standard vs. factored encoder)
Stage 13 : world scaling + severe out-of-distribution test

Stage 1 is validated (held-out generalization accuracy = 1.000, see README
for the reference run). Stages 2-12 are implemented and were validated in
prior runs; see README "Running" section for how to re-invoke any of them
individually.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import itertools
from collections import defaultdict

from config import (
    DEVICE, SHAPES, COLORS, SIZES, RELATIONS, HIDDEN_DIM, D,
    LR, BATCH_SIZE, GRAD_CLIP_NORM,
    EPOCHS_STAGE1, EPOCHS_STAGE2, EPOCHS_STAGE3, EPOCHS_STAGE4, EPOCHS_STAGE5,
    EPOCHS_STAGE6, EPOCHS_STAGE7_BASELINE, EPOCHS_STAGE8_MULTISPLIT,
    EPOCHS_STAGE9_ABLATION, EPOCHS_STAGE10_SEED_S3, EPOCHS_STAGE10_SEED_S5,
    EPOCHS_STAGE10_SEED_S6, EPOCHS_STAGE11_TRANSFORMER,
    N_PER_TRAIN_COMBO, N_PER_HELDOUT_COMBO,
    N_TRAIN_PAIRS_BASE, N_HELD_PAIRS_BASE,
    N_TRAIN_PARAGRAPHS, N_HELD_PARAGRAPHS,
    COMPOSITIONALITY_N_TRAIN_PAIRS, COMPOSITIONALITY_N_TEST_PAIRS,
    MULTI_SEED_LIST,
    WORLD_SCALING_CONFIGS, WORLD_SCALING_EPOCHS, WORLD_SCALING_BATCH_SIZE,
    WORLD_SCALING_N_TRAIN_PAIRS, WORLD_SCALING_N_TEST_PAIRS,
    WORLD_SCALING_N_TRAIN_PARAGRAPHS, WORLD_SCALING_N_TEST_PARAGRAPHS,
    set_all_seeds,
)

set_all_seeds()  # deterministic=False by default: matches original speed-oriented cudnn setting
print(f"Device: {DEVICE}")

N_SHAPES, N_COLORS, N_SIZES, N_RELATIONS = (
    len(SHAPES), len(COLORS), len(SIZES), len(RELATIONS)
)
OBJ_DIM = N_SHAPES + N_COLORS + N_SIZES
SCENE_DIM = OBJ_DIM * 2 + N_RELATIONS   # object1 + object2 + relation


# ─────────────────────────────────────────────────────
# SCENE ENCODING (shared across all stages)
# ─────────────────────────────────────────────────────
def encode_object(shape, color, size):
    s = F.one_hot(torch.tensor(SHAPES.index(shape)), N_SHAPES).float()
    c = F.one_hot(torch.tensor(COLORS.index(color)), N_COLORS).float()
    z = F.one_hot(torch.tensor(SIZES.index(size)), N_SIZES).float()
    return torch.cat([s, c, z])


def encode_scene(obj1, rel, obj2):
    o1 = encode_object(*obj1)
    o2 = encode_object(*obj2)
    r = F.one_hot(torch.tensor(RELATIONS.index(rel)), N_RELATIONS).float()
    return torch.cat([o1, r, o2])


def scene_to_sentence(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2
    return f"a {z1} {c1} {s1} is {rel} a {z2} {c2} {s2}"


# ─────────────────────────────────────────────────────
# TRAIN/HELD-OUT COMBO SPLITTING
# (shared across all stages, to keep comparisons fair)
# ─────────────────────────────────────────────────────
def all_objects():
    return list(itertools.product(SHAPES, COLORS, SIZES))

all_objs = all_objects()
random.shuffle(all_objs)

seen_s, seen_c, seen_z = set(), set(), set()
train_objs, extra_objs = [], []
for obj in all_objs:
    s, c, z = obj
    if s not in seen_s or c not in seen_c or z not in seen_z:
        train_objs.append(obj)
        seen_s.add(s); seen_c.add(c); seen_z.add(z)
    else:
        extra_objs.append(obj)

random.shuffle(extra_objs)
train_objs += extra_objs[:6]
held_out_objs = extra_objs[6:14]

def make_pairs(objs, n_pairs, exclude_relation_obj=None):
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 30:
        attempts += 1
        o1 = random.choice(objs)
        o2 = random.choice(objs)
        if o1 == o2:
            continue
        rel = random.choice(RELATIONS)
        if exclude_relation_obj and (rel, o1) in exclude_relation_obj:
            continue
        pairs.append((o1, rel, o2))
    return pairs

HELD_RELATION_SHAPE = {("above", "star"), ("below", "triangle")}
exclude_set = set()
for obj in train_objs + held_out_objs:
    for rel, shp in HELD_RELATION_SHAPE:
        if obj[0] == shp:
            exclude_set.add((rel, obj))

train_pairs = make_pairs(train_objs, N_TRAIN_PAIRS_BASE, exclude_relation_obj=exclude_set)

held_pairs_new_objects = make_pairs(held_out_objs, N_HELD_PAIRS_BASE)
held_pairs_relation_shape = []
attempts = 0
while len(held_pairs_relation_shape) < N_HELD_PAIRS_BASE and attempts < 500:
    attempts += 1
    rel, shp = random.choice(list(HELD_RELATION_SHAPE))
    matching_objs = [o for o in train_objs + held_out_objs if o[0] == shp]
    if not matching_objs:
        continue
    o1 = random.choice(matching_objs)
    o2 = random.choice(train_objs)
    if o1 == o2:
        continue
    held_pairs_relation_shape.append((o1, rel, o2))

held_pairs = held_pairs_new_objects + held_pairs_relation_shape

print(f"   Train objects: {len(train_objs)} | Held-out objects: {len(held_out_objs)}")
print(f"   Train pairs: {len(train_pairs)} | Held-out pairs: {len(held_pairs)}\n")


def make_dataset(pairs, n_per=3):
    data = []
    for (obj1, rel, obj2) in pairs:
        scene_vec = encode_scene(obj1, rel, obj2)
        sentence = scene_to_sentence(obj1, rel, obj2)
        for _ in range(n_per):
            data.append((scene_vec, obj1, rel, obj2, sentence))
    random.shuffle(data)
    return data

train_data = make_dataset(train_pairs, n_per=N_PER_TRAIN_COMBO)
held_data = make_dataset(held_pairs, n_per=N_PER_HELDOUT_COMBO)

print(f"   Total training sentences: {len(train_data)}")
print(f"   Total held-out test sentences: {len(held_data)}\n")


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Direct Classification (validated baseline: held-out acc = 1.000)
# ═══════════════════════════════════════════════════════════════
class ObjectEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        attr_dim = HIDDEN_DIM // 4
        self.shape_proj = nn.Linear(N_SHAPES, attr_dim)
        self.color_proj = nn.Linear(N_COLORS, attr_dim)
        self.size_proj = nn.Linear(N_SIZES, attr_dim)

    def forward(self, obj_vec):
        shape_part = obj_vec[..., :N_SHAPES]
        color_part = obj_vec[..., N_SHAPES:N_SHAPES + N_COLORS]
        size_part = obj_vec[..., N_SHAPES + N_COLORS:]
        s = self.shape_proj(shape_part)
        c = self.color_proj(color_part)
        z = self.size_proj(size_part)
        return s, c, z


class RelationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        attr_dim = HIDDEN_DIM // 4
        self.rel_proj = nn.Linear(N_RELATIONS, attr_dim)

    def forward(self, rel_vec):
        return self.rel_proj(rel_vec)


class FactoredDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        attr_dim = HIDDEN_DIM // 4
        self.size1_head = nn.Linear(attr_dim, N_SIZES)
        self.color1_head = nn.Linear(attr_dim, N_COLORS)
        self.shape1_head = nn.Linear(attr_dim, N_SHAPES)
        self.rel_head = nn.Linear(attr_dim, N_RELATIONS)
        self.size2_head = nn.Linear(attr_dim, N_SIZES)
        self.color2_head = nn.Linear(attr_dim, N_COLORS)
        self.shape2_head = nn.Linear(attr_dim, N_SHAPES)

    def forward(self, s1, c1, z1, r, s2, c2, z2):
        return (
            self.size1_head(z1), self.color1_head(c1), self.shape1_head(s1),
            self.rel_head(r),
            self.size2_head(z2), self.color2_head(c2), self.shape2_head(s2),
        )

    @torch.no_grad()
    def generate(self, s1, c1, z1, r, s2, c2, z2):
        outs = self.forward(
            s1.unsqueeze(0), c1.unsqueeze(0), z1.unsqueeze(0),
            r.unsqueeze(0),
            s2.unsqueeze(0), c2.unsqueeze(0), z2.unsqueeze(0),
        )
        size1 = SIZES[outs[0].argmax(-1).item()]
        color1 = COLORS[outs[1].argmax(-1).item()]
        shape1 = SHAPES[outs[2].argmax(-1).item()]
        rel = RELATIONS[outs[3].argmax(-1).item()]
        size2 = SIZES[outs[4].argmax(-1).item()]
        color2 = COLORS[outs[5].argmax(-1).item()]
        shape2 = SHAPES[outs[6].argmax(-1).item()]
        return f"a {size1} {color1} {shape1} is {rel} a {size2} {color2} {shape2}"


class GroundedRelationalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = ObjectEncoder()
        self.rel_encoder = RelationEncoder()
        self.decoder = FactoredDecoder()

        attr_dim = HIDDEN_DIM // 4
        self.aux_shape = nn.Linear(attr_dim, N_SHAPES)
        self.aux_color = nn.Linear(attr_dim, N_COLORS)
        self.aux_size = nn.Linear(attr_dim, N_SIZES)
        self.aux_rel = nn.Linear(attr_dim, N_RELATIONS)

    def encode(self, scene_vec):
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]

        s1, c1, z1 = self.obj_encoder(obj1_vec)
        r = self.rel_encoder(rel_vec)
        s2, c2, z2 = self.obj_encoder(obj2_vec)
        return s1, c1, z1, r, s2, c2, z2

    def forward(self, scene_vec):
        s1, c1, z1, r, s2, c2, z2 = self.encode(scene_vec)
        outs = self.decoder(s1, c1, z1, r, s2, c2, z2)
        return outs, (s1, c1, z1, r, s2, c2, z2)

    def auxiliary_loss(self, scene_vec, parts):
        s1, c1, z1, r, s2, c2, z2 = parts
        n_s, n_c = N_SHAPES, N_COLORS
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]

        shape1_t = obj1_vec[..., :n_s].argmax(-1)
        color1_t = obj1_vec[..., n_s:n_s + n_c].argmax(-1)
        size1_t = obj1_vec[..., n_s + n_c:].argmax(-1)
        rel_t = rel_vec.argmax(-1)
        shape2_t = obj2_vec[..., :n_s].argmax(-1)
        color2_t = obj2_vec[..., n_s:n_s + n_c].argmax(-1)
        size2_t = obj2_vec[..., n_s + n_c:].argmax(-1)

        loss = (
            F.cross_entropy(self.aux_shape(s1), shape1_t) +
            F.cross_entropy(self.aux_color(c1), color1_t) +
            F.cross_entropy(self.aux_size(z1), size1_t) +
            F.cross_entropy(self.aux_rel(r), rel_t) +
            F.cross_entropy(self.aux_shape(s2), shape2_t) +
            F.cross_entropy(self.aux_color(c2), color2_t) +
            F.cross_entropy(self.aux_size(z2), size2_t)
        )
        return loss

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        s1, c1, z1, r, s2, c2, z2 = self.encode(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(s1[0], c1[0], z1[0], r[0], s2[0], c2[0], z2[0])
        self.train()
        return result


def train_stage1():
    model = GroundedRelationalModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Stage 1 — Direct Classification")
    print(f"   ({len(train_data)} sentences, {EPOCHS_STAGE1} epochs)\n")

    for epoch in range(EPOCHS_STAGE1):
        random.shuffle(train_data)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)

            size1_t = torch.tensor([SIZES.index(b[1][2]) for b in batch], device=DEVICE)
            color1_t = torch.tensor([COLORS.index(b[1][1]) for b in batch], device=DEVICE)
            shape1_t = torch.tensor([SHAPES.index(b[1][0]) for b in batch], device=DEVICE)
            rel_t = torch.tensor([RELATIONS.index(b[2]) for b in batch], device=DEVICE)
            size2_t = torch.tensor([SIZES.index(b[3][2]) for b in batch], device=DEVICE)
            color2_t = torch.tensor([COLORS.index(b[3][1]) for b in batch], device=DEVICE)
            shape2_t = torch.tensor([SHAPES.index(b[3][0]) for b in batch], device=DEVICE)

            outs, parts = model(scenes)
            gen_loss = (
                F.cross_entropy(outs[0], size1_t) + F.cross_entropy(outs[1], color1_t) +
                F.cross_entropy(outs[2], shape1_t) + F.cross_entropy(outs[3], rel_t) +
                F.cross_entropy(outs[4], size2_t) + F.cross_entropy(outs[5], color2_t) +
                F.cross_entropy(outs[6], shape2_t)
            )
            aux_loss = model.auxiliary_loss(scenes, parts)
            loss = gen_loss + 0.3 * aux_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 70 == 0 or epoch == EPOCHS_STAGE1 - 1:
            train_acc = evaluate_exact_match_s1(model, train_data[:60])
            held_acc = evaluate_exact_match_s1(model, held_data)
            print(f"  epoch {epoch:3d} | train_acc={train_acc:.3f} | held_out_acc={held_acc:.3f}")

    print("\nStage 1 done.\n")
    return model


@torch.no_grad()
def evaluate_exact_match_s1(model, data):
    model.eval()
    correct = 0
    for scene_vec, obj1, rel, obj2, correct_sentence in data:
        generated = model.describe(scene_vec)
        if generated.strip() == correct_sentence.strip():
            correct += 1
    model.train()
    return correct / len(data)


# ═══════════════════════════════════════════════════════════════
# VOCABULARY — shared across Stages 2-13 (real word tokens, not categories)
# ═══════════════════════════════════════════════════════════════
SPECIAL = ["<pad>", "<sos>", "<eos>"]
FUNCTION = ["a", "the", "is", "has", "and", ".", "over", "under",
            "it", "above", "below", "left", "right", "of"]
CONTENT = SHAPES + COLORS + SIZES

WORDS = SPECIAL + FUNCTION + CONTENT
assert len(WORDS) == len(set(WORDS)), "duplicate word in vocabulary"

word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(WORDS)
PAD, SOS, EOS = word2id["<pad>"], word2id["<sos>"], word2id["<eos>"]

print(f"Shared vocabulary: {VOCAB_SIZE} words (stages 2-13)\n")


def pad_batch(token_lists):
    """Pad a list of variable-length token sequences into a single tensor."""
    max_len = max(len(t) for t in token_lists)
    padded = torch.full((len(token_lists), max_len), PAD, dtype=torch.long)
    for i, t in enumerate(token_lists):
        padded[i, :len(t)] = torch.tensor(t)
    return padded


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — Learned Word Embeddings
# ═══════════════════════════════════════════════════════════════
class Stage2SlotEncoder(nn.Module):
    """Same idea as ObjectEncoder/RelationEncoder from Stage 1, using the shared dim D."""
    def __init__(self):
        super().__init__()
        self.shape_proj = nn.Linear(N_SHAPES, D)
        self.color_proj = nn.Linear(N_COLORS, D)
        self.size_proj = nn.Linear(N_SIZES, D)
        self.rel_proj = nn.Linear(N_RELATIONS, D)

    def forward_obj(self, obj_vec):
        shape_part = obj_vec[..., :N_SHAPES]
        color_part = obj_vec[..., N_SHAPES:N_SHAPES + N_COLORS]
        size_part = obj_vec[..., N_SHAPES + N_COLORS:]
        return self.shape_proj(shape_part), self.color_proj(color_part), self.size_proj(size_part)

    def forward_rel(self, rel_vec):
        return self.rel_proj(rel_vec)


class Stage2Model(nn.Module):
    """Each slot predicts a word via weight tying with the embedding table."""
    def __init__(self):
        super().__init__()
        self.slot_encoder = Stage2SlotEncoder()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.size1_out = nn.Linear(D, D)
        self.color1_out = nn.Linear(D, D)
        self.shape1_out = nn.Linear(D, D)
        self.rel_out = nn.Linear(D, D)
        self.size2_out = nn.Linear(D, D)
        self.color2_out = nn.Linear(D, D)
        self.shape2_out = nn.Linear(D, D)

    def encode(self, scene_vec):
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]
        s1, c1, z1 = self.slot_encoder.forward_obj(obj1_vec)
        r = self.slot_encoder.forward_rel(rel_vec)
        s2, c2, z2 = self.slot_encoder.forward_obj(obj2_vec)
        return s1, c1, z1, r, s2, c2, z2

    def _logits(self, proj_layer, slot_vec):
        h = proj_layer(slot_vec)                      # (B, D)
        return h @ self.embed.weight.T                # (B, VOCAB_SIZE)

    def forward(self, scene_vec):
        s1, c1, z1, r, s2, c2, z2 = self.encode(scene_vec)
        logits = (
            self._logits(self.size1_out, z1),
            self._logits(self.color1_out, c1),
            self._logits(self.shape1_out, s1),
            self._logits(self.rel_out, r),
            self._logits(self.size2_out, z2),
            self._logits(self.color2_out, c2),
            self._logits(self.shape2_out, s2),
        )
        return logits

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        logits = self.forward(scene_vec.unsqueeze(0).to(DEVICE))
        words = [id2word[lg.argmax(-1).item()] for lg in logits]
        size1, color1, shape1, rel, size2, color2, shape2 = words
        if rel in ("left", "right"):
            rel = f"{rel} of"
        self.train()
        return f"a {size1} {color1} {shape1} is {rel} a {size2} {color2} {shape2}"


def train_stage2():
    model = Stage2Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Stage 2 — Learned Word Embeddings (weight tying)")
    print(f"   ({len(train_data)} sentences, {EPOCHS_STAGE2} epochs)\n")

    for epoch in range(EPOCHS_STAGE2):
        random.shuffle(train_data)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)

            size1_t = torch.tensor([word2id[b[1][2]] for b in batch], device=DEVICE)
            color1_t = torch.tensor([word2id[b[1][1]] for b in batch], device=DEVICE)
            shape1_t = torch.tensor([word2id[b[1][0]] for b in batch], device=DEVICE)
            rel_first_word = [b[2].split()[0] for b in batch]
            rel_t = torch.tensor([word2id[w] for w in rel_first_word], device=DEVICE)
            size2_t = torch.tensor([word2id[b[3][2]] for b in batch], device=DEVICE)
            color2_t = torch.tensor([word2id[b[3][1]] for b in batch], device=DEVICE)
            shape2_t = torch.tensor([word2id[b[3][0]] for b in batch], device=DEVICE)

            logits = model(scenes)
            loss = (
                F.cross_entropy(logits[0], size1_t) + F.cross_entropy(logits[1], color1_t) +
                F.cross_entropy(logits[2], shape1_t) + F.cross_entropy(logits[3], rel_t) +
                F.cross_entropy(logits[4], size2_t) + F.cross_entropy(logits[5], color2_t) +
                F.cross_entropy(logits[6], shape2_t)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 70 == 0 or epoch == EPOCHS_STAGE2 - 1:
            train_acc = evaluate_exact_match_s2(model, train_data[:60])
            held_acc = evaluate_exact_match_s2(model, held_data)
            print(f"  epoch {epoch:3d} | train_acc={train_acc:.3f} | held_out_acc={held_acc:.3f}")

    print("\nStage 2 done.\n")
    return model


@torch.no_grad()
def evaluate_exact_match_s2(model, data):
    model.eval()
    correct = 0
    for scene_vec, obj1, rel, obj2, correct_sentence in data:
        generated = model.describe(scene_vec)
        if generated.strip() == correct_sentence.strip():
            correct += 1
    model.train()
    return correct / len(data)


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — Full Sentences, No Concept Labels (Seq2Seq + Attention)
# ═══════════════════════════════════════════════════════════════
def tokenize_canonical(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2
    words = ["the", z1, c1, s1, "is"] + rel.split() + ["the", z2, c2, s2, "."]
    return [SOS] + [word2id[w] for w in words] + [EOS]


def tokenize_canonical_with_attn(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2

    words = []
    attn_targets = []

    words.append("the")
    words.append(z1); attn_targets.append((len(words) - 1, 0))
    words.append(c1); attn_targets.append((len(words) - 1, 1))
    words.append(s1); attn_targets.append((len(words) - 1, 2))
    words.append("is")
    for w in rel.split():
        words.append(w); attn_targets.append((len(words) - 1, 3))
    words.append("the")
    words.append(z2); attn_targets.append((len(words) - 1, 4))
    words.append(c2); attn_targets.append((len(words) - 1, 5))
    words.append(s2); attn_targets.append((len(words) - 1, 6))
    words.append(".")

    tokens = [SOS] + [word2id[w] for w in words] + [EOS]
    return tokens, attn_targets


def sentence_text_canonical(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2
    return f"the {z1} {c1} {s1} is {rel} the {z2} {c2} {s2} ."


class Stage3SlotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_proj = nn.Linear(N_SHAPES, D)
        self.color_proj = nn.Linear(N_COLORS, D)
        self.size_proj = nn.Linear(N_SIZES, D)
        self.rel_proj = nn.Linear(N_RELATIONS, D)
        self.slot_id_embed = nn.Embedding(14, D)

        self.aux_shape = nn.Linear(D, N_SHAPES)
        self.aux_color = nn.Linear(D, N_COLORS)
        self.aux_size = nn.Linear(D, N_SIZES)
        self.aux_rel = nn.Linear(D, N_RELATIONS)

    def forward(self, scene_vec, slot_offset=0, return_raw=False, use_identity=True):
        """
        use_identity=False disables the slot identity embedding without
        touching the rest of the code — used by the "no_identity" ablation
        to test the attribute-binding mechanism.
        """
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]

        def obj_slots(obj_vec):
            sh = obj_vec[..., :N_SHAPES]
            co = obj_vec[..., N_SHAPES:N_SHAPES + N_COLORS]
            sz = obj_vec[..., N_SHAPES + N_COLORS:]
            return self.size_proj(sz), self.color_proj(co), self.shape_proj(sh)

        z1, c1, s1 = obj_slots(obj1_vec)
        r = self.rel_proj(rel_vec)
        z2, c2, s2 = obj_slots(obj2_vec)

        raw_slots = [z1, c1, s1, r, z2, c2, s2]

        if use_identity:
            slot_ids = torch.arange(slot_offset, slot_offset + 7, device=scene_vec.device)
            id_vecs = self.slot_id_embed(slot_ids)          # (7, D)
            slots = [raw_slots[i] + id_vecs[i].unsqueeze(0) for i in range(7)]
        else:
            slots = raw_slots

        memory = torch.stack(slots, dim=1)     # (B, 7, D)
        if return_raw:
            return memory, raw_slots
        return memory

    def auxiliary_loss(self, scene_vec, raw_slots):
        z1, c1, s1, r, z2, c2, s2 = raw_slots
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]

        shape1_t = obj1_vec[..., :N_SHAPES].argmax(-1)
        color1_t = obj1_vec[..., N_SHAPES:N_SHAPES + N_COLORS].argmax(-1)
        size1_t = obj1_vec[..., N_SHAPES + N_COLORS:].argmax(-1)
        rel_t = rel_vec.argmax(-1)
        shape2_t = obj2_vec[..., :N_SHAPES].argmax(-1)
        color2_t = obj2_vec[..., N_SHAPES:N_SHAPES + N_COLORS].argmax(-1)
        size2_t = obj2_vec[..., N_SHAPES + N_COLORS:].argmax(-1)

        loss = (
            F.cross_entropy(self.aux_size(z1), size1_t) +
            F.cross_entropy(self.aux_color(c1), color1_t) +
            F.cross_entropy(self.aux_shape(s1), shape1_t) +
            F.cross_entropy(self.aux_rel(r), rel_t) +
            F.cross_entropy(self.aux_size(z2), size2_t) +
            F.cross_entropy(self.aux_color(c2), color2_t) +
            F.cross_entropy(self.aux_shape(s2), shape2_t)
        )
        return loss


class AttentionDecoder(nn.Module):
    def __init__(self, embed_table):
        super().__init__()
        self.embed = embed_table
        self.lstm_cell = nn.LSTMCell(D * 2, D)
        self.out_proj = nn.Linear(D, D)

    def _step(self, prev_token_ids, h, c, memory):
        emb = self.embed(prev_token_ids)
        scores = torch.bmm(memory, h.unsqueeze(2)).squeeze(2)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
        lstm_in = torch.cat([emb, context], dim=1)
        h, c = self.lstm_cell(lstm_in, (h, c))
        logits = self.out_proj(h) @ self.embed.weight.T
        return logits, h, c, attn

    def forward(self, memory, target_tokens, return_attn=False):
        B = memory.size(0)
        h = memory.mean(dim=1)
        c = torch.zeros_like(h)

        all_logits, all_attns = [], []
        for t in range(target_tokens.size(1) - 1):
            prev = target_tokens[:, t]
            logits, h, c, attn = self._step(prev, h, c, memory)
            all_logits.append(logits)
            all_attns.append(attn)
        logits_out = torch.stack(all_logits, dim=1)
        if return_attn:
            return logits_out, torch.stack(all_attns, dim=1)
        return logits_out

    @torch.no_grad()
    def generate(self, memory, max_len=14):
        B = memory.size(0)
        h = memory.mean(dim=1)
        c = torch.zeros_like(h)
        token = torch.full((B,), SOS, dtype=torch.long, device=memory.device)

        words_per_batch = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            logits, h, c, _attn = self._step(token, h, c, memory)
            token = logits.argmax(-1)
            for b in range(B):
                if done[b]:
                    continue
                tid = token[b].item()
                if tid == EOS:
                    done[b] = True
                else:
                    words_per_batch[b].append(id2word.get(tid, "<unk>"))
            if all(done):
                break
        return [" ".join(w) for w in words_per_batch]


class Stage3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = Stage3SlotEncoder()
        self.decoder = AttentionDecoder(self.embed)

    def forward(self, scene_vec, target_tokens, return_attn=False):
        memory, raw_slots = self.slot_encoder(scene_vec, return_raw=True)
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw_slots, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw_slots

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        memory = self.slot_encoder(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(memory)[0]
        self.train()
        return result


def train_stage3(epochs=EPOCHS_STAGE3):
    model = Stage3Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def build_s3_data(pairs, n_per):
        data = []
        for (obj1, rel, obj2) in pairs:
            scene_vec = encode_scene(obj1, rel, obj2)
            tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
            text = sentence_text_canonical(obj1, rel, obj2)
            for _ in range(n_per):
                data.append((scene_vec, tokens, attn_targets, text))
        random.shuffle(data)
        return data

    s3_train = build_s3_data(train_pairs, N_PER_TRAIN_COMBO)
    s3_held = build_s3_data(held_pairs, N_PER_HELDOUT_COMBO)

    print("Stage 3 — Full Sentence Generation (no concept labels)")
    print(f"   ({len(s3_train)} sentences, {epochs} epochs)\n")

    for epoch in range(epochs):
        random.shuffle(s3_train)
        total_loss, n_batches = 0.0, 0
        use_aux = epoch >= 60
        for i in range(0, len(s3_train), BATCH_SIZE):
            batch = s3_train[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)
            tokens = pad_batch([b[1] for b in batch]).to(DEVICE)
            batch_attn_targets = [b[2] for b in batch]

            logits, raw_slots, attn = model(scenes, tokens, return_attn=True)
            target = tokens[:, 1:]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1),
                ignore_index=PAD
            )

            attn_terms = []
            for b_idx, targets_list in enumerate(batch_attn_targets):
                for (pos, slot) in targets_list:
                    if pos < attn.size(1):
                        probs = attn[b_idx, pos, :].unsqueeze(0)
                        target_slot = torch.tensor([slot], device=DEVICE)
                        attn_terms.append(F.nll_loss(torch.log(probs + 1e-8), target_slot))
            attn_sup_loss = torch.stack(attn_terms).mean() if attn_terms else torch.tensor(0.0, device=DEVICE)

            if use_aux:
                aux_loss = model.slot_encoder.auxiliary_loss(scenes, raw_slots)
                loss = gen_loss + 0.15 * aux_loss + 0.5 * attn_sup_loss
            else:
                loss = gen_loss + 0.5 * attn_sup_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        if epoch % 50 == 0 or epoch == epochs - 1:
            train_acc = evaluate_exact_match_s3(model, s3_train[:60])
            held_acc = evaluate_exact_match_s3(model, s3_held)
            print(f"  epoch {epoch:3d} | loss={total_loss/n_batches:.3f} | "
                  f"train_acc={train_acc:.3f} | held_out_acc={held_acc:.3f}")

    print("\nStage 3 done.\n")

    print("=" * 70)
    print("Stage 3 — per-attribute accuracy breakdown")
    print("=" * 70)
    attr_acc = evaluate_per_attribute_s3(model, s3_held)
    for attr, acc in attr_acc.items():
        print(f"  {attr:10s}: {acc:.3f}")
    print()

    return model, s3_train, s3_held


@torch.no_grad()
def evaluate_exact_match_s3(model, data):
    model.eval()
    correct = 0
    for scene_vec, tokens, attn_targets, correct_text in data:
        generated = model.describe(scene_vec)
        if generated.strip() == correct_text.strip():
            correct += 1
    model.train()
    return correct / len(data)


@torch.no_grad()
def evaluate_per_attribute_s3(model, data):
    model.eval()
    attr_names = {0: "size1", 1: "color1", 2: "shape1",
                  4: "size2", 5: "color2", 6: "shape2"}
    correct = defaultdict(int)
    total = defaultdict(int)

    for scene_vec, tokens, attn_targets, correct_text in data:
        generated = model.describe(scene_vec)
        gen_words = generated.split()
        correct_words = correct_text.split()

        for (pos, slot) in attn_targets:
            if slot not in attr_names:
                continue
            attr = attr_names[slot]
            total[attr] += 1
            if pos < len(gen_words) and pos < len(correct_words):
                if gen_words[pos] == correct_words[pos]:
                    correct[attr] += 1

    model.train()
    result = {attr: (correct[attr] / total[attr] if total[attr] > 0 else 0.0)
              for attr in total}
    grouped = defaultdict(lambda: [0, 0])
    for attr in total:
        base = attr[:-1]
        grouped[base][0] += correct[attr]
        grouped[base][1] += total[attr]
    for base, (c, t) in grouped.items():
        result[f"{base} (overall)"] = c / t if t > 0 else 0.0
    return result


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — Paraphrase Understanding (Contrastive Alignment)
# ═══════════════════════════════════════════════════════════════
REL_OVER_WORD = {
    "above": ["over"],
    "below": ["under"],
    "left of": ["to", "the", "left", "of"],
    "right of": ["to", "the", "right", "of"],
}
_new_words = ["to"]
for w in _new_words:
    if w not in word2id:
        word2id[w] = len(WORDS)
        id2word[word2id[w]] = w
        WORDS.append(w)
VOCAB_SIZE = len(WORDS)


def make_paraphrases(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2

    canonical = ["the", z1, c1, s1, "is"] + rel.split() + ["the", z2, c2, s2, "."]
    fronted = rel.split() + ["the", z2, c2, s2, "is", "the", z1, c1, s1, "."]
    has_cons = ["the", z2, c2, s2, "has", "the", z1, c1, s1] + REL_OVER_WORD[rel] + ["it", "."]

    return [canonical, fronted, has_cons]


class Stage4SentenceEncoder(nn.Module):
    def __init__(self, embed_table):
        super().__init__()
        self.embed = embed_table
        self.lstm = nn.LSTM(D, D, batch_first=True)

    def forward(self, token_batch):
        emb = self.embed(token_batch)
        _, (h, _) = self.lstm(emb)
        vec = h[-1]
        return F.normalize(vec, dim=-1)


class Stage4SceneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_encoder = Stage3SlotEncoder()

    def forward(self, scene_vec):
        memory = self.slot_encoder(scene_vec)
        vec = memory.mean(dim=1)
        return F.normalize(vec, dim=-1)


class Stage4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.sentence_encoder = Stage4SentenceEncoder(self.embed)
        self.scene_encoder = Stage4SceneEncoder()
        self.logit_scale = 10.0

    def forward(self, scene_vecs, token_batches):
        text_vecs = self.sentence_encoder(token_batches)
        scene_vecs = self.scene_encoder(scene_vecs)
        sim = text_vecs @ scene_vecs.T * self.logit_scale
        return sim

    @torch.no_grad()
    def encode_sentence_text(self, tokens):
        self.eval()
        t = torch.tensor(tokens, device=DEVICE).unsqueeze(0)
        vec = self.sentence_encoder(t)[0]
        self.train()
        return vec


def train_stage4(epochs=EPOCHS_STAGE4):
    model = Stage4Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    s4_facts = train_pairs[:]

    print("Stage 4 — Paraphrase Understanding (contrastive)")
    print(f"   ({len(s4_facts)} facts x 3 paraphrases each, {epochs} epochs)\n")

    for epoch in range(epochs):
        random.shuffle(s4_facts)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(s4_facts), BATCH_SIZE):
            batch_facts = s4_facts[i:i + BATCH_SIZE]
            if len(batch_facts) < 2:
                continue

            scene_list, token_list = [], []
            for (obj1, rel, obj2) in batch_facts:
                scene_list.append(encode_scene(obj1, rel, obj2))
                paraphrases = make_paraphrases(obj1, rel, obj2)
                chosen = random.choice(paraphrases)
                token_list.append([SOS] + [word2id[w] for w in chosen] + [EOS])

            scenes = torch.stack(scene_list).to(DEVICE)
            tokens = pad_batch(token_list).to(DEVICE)

            sim = model(scenes, tokens)
            labels = torch.arange(sim.size(0), device=DEVICE)
            loss_t2s = F.cross_entropy(sim, labels)
            loss_s2t = F.cross_entropy(sim.T, labels)
            loss = (loss_t2s + loss_s2t) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1

        if epoch % 40 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(n_batches, 1)
            para_acc = evaluate_paraphrase_recognition(model, held_pairs[:15])
            print(f"  epoch {epoch:3d} | loss={avg_loss:.3f} | "
                  f"paraphrase_recognition_acc={para_acc:.3f}")

    print("\nStage 4 done.\n")
    return model


@torch.no_grad()
def evaluate_paraphrase_recognition(model, facts, n_distractors=4):
    model.eval()
    if len(facts) < n_distractors + 1:
        model.train()
        return 0.0

    correct = 0
    total = 0
    for (obj1, rel, obj2) in facts:
        paraphrases = make_paraphrases(obj1, rel, obj2)
        anchor_tokens = [SOS] + [word2id[w] for w in paraphrases[0]] + [EOS]
        positive_tokens = [SOS] + [word2id[w] for w in paraphrases[1]] + [EOS]

        anchor_vec = model.encode_sentence_text(anchor_tokens)
        positive_vec = model.encode_sentence_text(positive_tokens)
        positive_sim = F.cosine_similarity(anchor_vec.unsqueeze(0), positive_vec.unsqueeze(0)).item()

        distractor_facts = random.sample(
            [f for f in facts if f != (obj1, rel, obj2)],
            min(n_distractors, len(facts) - 1)
        )
        distractor_sims = []
        for d_obj1, d_rel, d_obj2 in distractor_facts:
            d_paraphrases = make_paraphrases(d_obj1, d_rel, d_obj2)
            d_tokens = [SOS] + [word2id[w] for w in d_paraphrases[1]] + [EOS]
            d_vec = model.encode_sentence_text(d_tokens)
            sim = F.cosine_similarity(anchor_vec.unsqueeze(0), d_vec.unsqueeze(0)).item()
            distractor_sims.append(sim)

        if positive_sim > max(distractor_sims, default=-1):
            correct += 1
        total += 1

    model.train()
    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — Paragraphs (two facts -> a two-sentence paragraph)
# ═══════════════════════════════════════════════════════════════
def make_paragraph_fact_pair():
    fact1 = random.choice(train_pairs)
    fact2 = random.choice(train_pairs)
    return fact1, fact2


def tokenize_paragraph(fact1, fact2):
    obj1_a, rel_a, obj2_a = fact1
    obj1_b, rel_b, obj2_b = fact2
    s1a, c1a, z1a = obj1_a; s2a, c2a, z2a = obj2_a
    s1b, c1b, z1b = obj1_b; s2b, c2b, z2b = obj2_b

    sent1 = ["the", z1a, c1a, s1a, "is"] + rel_a.split() + ["the", z2a, c2a, s2a, "."]
    sent2 = ["the", z1b, c1b, s1b, "is"] + rel_b.split() + ["the", z2b, c2b, s2b, "."]
    words = sent1 + sent2
    return [SOS] + [word2id[w] for w in words] + [EOS]


def tokenize_paragraph_with_attn(fact1, fact2):
    obj1_a, rel_a, obj2_a = fact1
    obj1_b, rel_b, obj2_b = fact2
    s1a, c1a, z1a = obj1_a; s2a, c2a, z2a = obj2_a
    s1b, c1b, z1b = obj1_b; s2b, c2b, z2b = obj2_b

    words = []
    attn_targets = []

    def add(word, slot=None):
        words.append(word)
        if slot is not None:
            attn_targets.append((len(words) - 1, slot))

    add("the")
    add(z1a, 0); add(c1a, 1); add(s1a, 2)
    add("is")
    for w in rel_a.split():
        add(w, 3)
    add("the")
    add(z2a, 4); add(c2a, 5); add(s2a, 6)
    add(".")

    add("the")
    add(z1b, 7); add(c1b, 8); add(s1b, 9)
    add("is")
    for w in rel_b.split():
        add(w, 10)
    add("the")
    add(z2b, 11); add(c2b, 12); add(s2b, 13)
    add(".")

    tokens = [SOS] + [word2id[w] for w in words] + [EOS]
    return tokens, attn_targets


def paragraph_text(fact1, fact2):
    obj1_a, rel_a, obj2_a = fact1
    obj1_b, rel_b, obj2_b = fact2
    s1a, c1a, z1a = obj1_a; s2a, c2a, z2a = obj2_a
    s1b, c1b, z1b = obj1_b; s2b, c2b, z2b = obj2_b
    return (f"the {z1a} {c1a} {s1a} is {rel_a} the {z2a} {c2a} {s2a} . "
            f"the {z1b} {c1b} {s1b} is {rel_b} the {z2b} {c2b} {s2b} .")


class Stage5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = Stage3SlotEncoder()
        self.decoder = AttentionDecoder(self.embed)

    def encode_two_facts(self, scene_vec1, scene_vec2):
        memory1, raw1 = self.slot_encoder(scene_vec1, slot_offset=0, return_raw=True)
        memory2, raw2 = self.slot_encoder(scene_vec2, slot_offset=7, return_raw=True)
        memory = torch.cat([memory1, memory2], dim=1)
        return memory, raw1, raw2

    def forward(self, scene_vec1, scene_vec2, target_tokens, return_attn=False):
        memory, raw1, raw2 = self.encode_two_facts(scene_vec1, scene_vec2)
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw1, raw2, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw1, raw2

    @torch.no_grad()
    def describe(self, scene_vec1, scene_vec2):
        self.eval()
        memory, _, _ = self.encode_two_facts(
            scene_vec1.unsqueeze(0).to(DEVICE),
            scene_vec2.unsqueeze(0).to(DEVICE),
        )
        result = self.decoder.generate(memory, max_len=26)[0]
        self.train()
        return result


def train_stage5(epochs=EPOCHS_STAGE5, n_train_paragraphs=N_TRAIN_PARAGRAPHS,
                  n_held_paragraphs=N_HELD_PARAGRAPHS):
    model = Stage5Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def build_s5_data(fact_pool, n_paragraphs):
        data = []
        for _ in range(n_paragraphs):
            fact1 = random.choice(fact_pool)
            fact2 = random.choice(fact_pool)
            sv1 = encode_scene(*fact1)
            sv2 = encode_scene(*fact2)
            tokens, attn_targets = tokenize_paragraph_with_attn(fact1, fact2)
            text = paragraph_text(fact1, fact2)
            data.append((sv1, sv2, tokens, attn_targets, text))
        return data

    s5_train = build_s5_data(train_pairs, n_train_paragraphs)
    s5_held = build_s5_data(held_pairs, n_held_paragraphs)

    print("Stage 5 — Paragraphs (two linked facts)")
    print(f"   ({len(s5_train)} paragraphs, {epochs} epochs)\n")

    for epoch in range(epochs):
        random.shuffle(s5_train)
        total_loss, n_batches = 0.0, 0
        use_aux = epoch >= 60
        for i in range(0, len(s5_train), BATCH_SIZE):
            batch = s5_train[i:i + BATCH_SIZE]
            sv1 = torch.stack([b[0] for b in batch]).to(DEVICE)
            sv2 = torch.stack([b[1] for b in batch]).to(DEVICE)
            tokens = pad_batch([b[2] for b in batch]).to(DEVICE)
            batch_attn_targets = [b[3] for b in batch]

            logits, raw1, raw2, attn = model(sv1, sv2, tokens, return_attn=True)
            target = tokens[:, 1:]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1),
                ignore_index=PAD
            )

            attn_terms = []
            for b_idx, targets_list in enumerate(batch_attn_targets):
                for (pos, slot) in targets_list:
                    if pos < attn.size(1):
                        probs = attn[b_idx, pos, :].unsqueeze(0)
                        target_slot = torch.tensor([slot], device=DEVICE)
                        attn_terms.append(F.nll_loss(torch.log(probs + 1e-8), target_slot))
            attn_sup_loss = torch.stack(attn_terms).mean() if attn_terms else torch.tensor(0.0, device=DEVICE)

            if use_aux:
                aux_loss = (model.slot_encoder.auxiliary_loss(sv1, raw1) +
                            model.slot_encoder.auxiliary_loss(sv2, raw2))
                loss = gen_loss + 0.15 * aux_loss + 0.5 * attn_sup_loss
            else:
                loss = gen_loss + 0.5 * attn_sup_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        if epoch % 50 == 0 or epoch == epochs - 1:
            train_acc = evaluate_exact_match_s5(model, s5_train[:40])
            held_acc = evaluate_exact_match_s5(model, s5_held)
            print(f"  epoch {epoch:3d} | loss={total_loss/n_batches:.3f} | "
                  f"train_acc={train_acc:.3f} | held_out_acc={held_acc:.3f}")

    print("\nStage 5 done.\n")

    print("=" * 70)
    print("Stage 5 — per-attribute accuracy breakdown")
    print("=" * 70)
    attr_acc = evaluate_per_attribute_s5(model, s5_held)
    for attr, acc in attr_acc.items():
        print(f"  {attr:18s}: {acc:.3f}")
    print()

    return model, s5_train, s5_held


@torch.no_grad()
def evaluate_per_attribute_s5(model, data):
    model.eval()
    attr_names = {
        0: "size1_f1", 1: "color1_f1", 2: "shape1_f1",
        4: "size2_f1", 5: "color2_f1", 6: "shape2_f1",
        7: "size1_f2", 8: "color1_f2", 9: "shape1_f2",
        11: "size2_f2", 12: "color2_f2", 13: "shape2_f2",
    }
    correct = defaultdict(int)
    total = defaultdict(int)

    for sv1, sv2, tokens, attn_targets, correct_text in data:
        generated = model.describe(sv1, sv2)
        gen_words = generated.split()
        correct_words = correct_text.split()

        for (pos, slot) in attn_targets:
            if slot not in attr_names:
                continue
            attr = attr_names[slot]
            total[attr] += 1
            if pos < len(gen_words) and pos < len(correct_words):
                if gen_words[pos] == correct_words[pos]:
                    correct[attr] += 1

    model.train()
    result = {}
    grouped = defaultdict(lambda: [0, 0])
    for attr in total:
        base = attr.split("_")[0][:-1]
        grouped[base][0] += correct[attr]
        grouped[base][1] += total[attr]
    for base, (c, t) in grouped.items():
        result[f"{base} (overall)"] = c / t if t > 0 else 0.0
    return result


@torch.no_grad()
def evaluate_exact_match_s5(model, data):
    model.eval()
    correct = 0
    for sv1, sv2, tokens, attn_targets, correct_text in data:
        generated = model.describe(sv1, sv2)
        if generated.strip() == correct_text.strip():
            correct += 1
    model.train()
    return correct / len(data)


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — Compositional Generalization Test (single split)
# ═══════════════════════════════════════════════════════════════
def build_compositionality_split():
    HELD_RELATION = "above"
    HELD_SHAPE_AS_OBJ1 = "circle"

    comp_train_pairs, comp_test_pairs = [], []
    all_objs_comp = all_objects()

    attempts = 0
    while len(comp_train_pairs) < COMPOSITIONALITY_N_TRAIN_PAIRS and attempts < 5000:
        attempts += 1
        o1 = random.choice(all_objs_comp)
        o2 = random.choice(all_objs_comp)
        if o1 == o2:
            continue
        rel = random.choice(RELATIONS)
        if rel == HELD_RELATION and o1[0] == HELD_SHAPE_AS_OBJ1:
            continue
        comp_train_pairs.append((o1, rel, o2))

    attempts = 0
    while len(comp_test_pairs) < COMPOSITIONALITY_N_TEST_PAIRS and attempts < 3000:
        attempts += 1
        o1 = random.choice(all_objs_comp)
        o2 = random.choice(all_objs_comp)
        if o1 == o2:
            continue
        if o1[0] != HELD_SHAPE_AS_OBJ1:
            continue
        comp_test_pairs.append((o1, HELD_RELATION, o2))

    return comp_train_pairs, comp_test_pairs, HELD_RELATION, HELD_SHAPE_AS_OBJ1


def train_compositionality_test(epochs=EPOCHS_STAGE6):
    comp_train_pairs, comp_test_pairs, held_rel, held_shape = build_compositionality_split()

    print(f"   Held-out combination: relation='{held_rel}' + obj1.shape='{held_shape}'")
    print(f"   (this exact combination is 100% absent from training)")
    print(f"   Train pairs: {len(comp_train_pairs)} | Test pairs: {len(comp_test_pairs)}\n")

    rel_seen_elsewhere = sum(1 for (o1, r, o2) in comp_train_pairs if r == held_rel)
    shape_seen_elsewhere = sum(1 for (o1, r, o2) in comp_train_pairs if o1[0] == held_shape)
    print(f"   '{held_rel}' appears {rel_seen_elsewhere} times (with other shapes)")
    print(f"   '{held_shape}' as obj1 appears {shape_seen_elsewhere} times (with other relations)\n")

    model = Stage3Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def build_data(pairs, n_per):
        data = []
        for (obj1, rel, obj2) in pairs:
            scene_vec = encode_scene(obj1, rel, obj2)
            tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
            text = sentence_text_canonical(obj1, rel, obj2)
            for _ in range(n_per):
                data.append((scene_vec, tokens, attn_targets, text))
        random.shuffle(data)
        return data

    comp_train_data = build_data(comp_train_pairs, 3)
    comp_test_data = build_data(comp_test_pairs, 2)

    for epoch in range(epochs):
        random.shuffle(comp_train_data)
        use_aux = epoch >= 60
        for i in range(0, len(comp_train_data), BATCH_SIZE):
            batch = comp_train_data[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)
            tokens = pad_batch([b[1] for b in batch]).to(DEVICE)
            batch_attn_targets = [b[2] for b in batch]

            logits, raw_slots, attn = model(scenes, tokens, return_attn=True)
            target = tokens[:, 1:]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )
            attn_terms = []
            for b_idx, targets_list in enumerate(batch_attn_targets):
                for (pos, slot) in targets_list:
                    if pos < attn.size(1):
                        probs = attn[b_idx, pos, :].unsqueeze(0)
                        attn_terms.append(F.nll_loss(
                            torch.log(probs + 1e-8),
                            torch.tensor([slot], device=DEVICE)
                        ))
            attn_sup_loss = torch.stack(attn_terms).mean() if attn_terms else torch.tensor(0.0, device=DEVICE)

            if use_aux:
                aux_loss = model.slot_encoder.auxiliary_loss(scenes, raw_slots)
                loss = gen_loss + 0.15 * aux_loss + 0.5 * attn_sup_loss
            else:
                loss = gen_loss + 0.5 * attn_sup_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            train_acc = evaluate_exact_match_s3(model, comp_train_data[:60])
            test_acc = evaluate_exact_match_s3(model, comp_test_data)
            print(f"  epoch {epoch:3d} | train_acc={train_acc:.3f} | "
                  f"compositional_test_acc={test_acc:.3f}")

    print("\nCompositionality test training done.\n")
    print("=" * 70)
    print("Examples from the compositional test (never-seen combination)")
    print("=" * 70)
    for scene_vec, tokens, attn_targets, correct_text in comp_test_data[:8]:
        gen = model.describe(scene_vec)
        ok = "OK" if gen.strip() == correct_text.strip() else "MISS"
        print(f"  correct  : {correct_text}")
        print(f"  generated: {gen}  [{ok}]\n")

    final_acc = evaluate_exact_match_s3(model, comp_test_data)
    print(f"Final compositional generalization accuracy: {final_acc:.3f}")
    if final_acc > 0.5:
        print("Strong evidence of concept recombination (compositionality).")
    elif final_acc > 0.2:
        print("Partial generalization — the model attempts recombination but is not fully stable.")
    else:
        print("Still relying more on familiar combinations than true recombination.")
    return model, final_acc


# ═══════════════════════════════════════════════════════════════
# STAGE 7 — GRU Baseline
# ═══════════════════════════════════════════════════════════════
class GRUBaselineEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, D), nn.ReLU(),
            nn.Linear(D, D)
        )

    def forward(self, scene_vec):
        return self.net(scene_vec)


class GRUBaselineDecoder(nn.Module):
    def __init__(self, embed_table):
        super().__init__()
        self.embed = embed_table
        self.gru = nn.GRU(D, D, batch_first=True)
        self.out_proj = nn.Linear(D, D)

    def forward(self, scene_embed, target_tokens):
        h0 = scene_embed.unsqueeze(0)
        emb = self.embed(target_tokens[:, :-1])
        out, _ = self.gru(emb, h0)
        logits = self.out_proj(out) @ self.embed.weight.T
        return logits

    @torch.no_grad()
    def generate(self, scene_embed, max_len=14):
        B = scene_embed.size(0)
        h = scene_embed.unsqueeze(0)
        token = torch.full((B,), SOS, dtype=torch.long, device=scene_embed.device)
        words_per_batch = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            emb = self.embed(token).unsqueeze(1)
            out, h = self.gru(emb, h)
            logits = self.out_proj(out[:, -1]) @ self.embed.weight.T
            token = logits.argmax(-1)
            for b in range(B):
                if done[b]:
                    continue
                tid = token[b].item()
                if tid == EOS:
                    done[b] = True
                else:
                    words_per_batch[b].append(id2word.get(tid, "<unk>"))
            if all(done):
                break
        return [" ".join(w) for w in words_per_batch]


class GRUBaselineModel(nn.Module):
    def __init__(self, input_dim=SCENE_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.encoder = GRUBaselineEncoder(input_dim)
        self.decoder = GRUBaselineDecoder(self.embed)

    def forward(self, scene_vec, target_tokens):
        scene_embed = self.encoder(scene_vec)
        return self.decoder(scene_embed, target_tokens)

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        scene_embed = self.encoder(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(scene_embed)[0]
        self.train()
        return result


class GRUBaselineModelDual(nn.Module):
    def __init__(self, single_scene_dim=SCENE_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.encoder = GRUBaselineEncoder(single_scene_dim * 2)
        self.decoder = GRUBaselineDecoder(self.embed)

    def forward(self, sv1, sv2, target_tokens):
        combined = torch.cat([sv1, sv2], dim=-1)
        scene_embed = self.encoder(combined)
        return self.decoder(scene_embed, target_tokens)

    @torch.no_grad()
    def describe(self, sv1, sv2):
        self.eval()
        combined = torch.cat([sv1.unsqueeze(0), sv2.unsqueeze(0)], dim=-1).to(DEVICE)
        scene_embed = self.encoder(combined)
        result = self.decoder.generate(scene_embed)[0]
        self.train()
        return result


def _train_baseline_generic(model, train_data, held_data, epochs, is_dual=False, label=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        random.shuffle(train_data)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            tokens_idx = 2 if is_dual else 1
            tokens = pad_batch([b[tokens_idx] for b in batch]).to(DEVICE)

            if is_dual:
                sv1 = torch.stack([b[0] for b in batch]).to(DEVICE)
                sv2 = torch.stack([b[1] for b in batch]).to(DEVICE)
                logits = model(sv1, sv2, tokens)
            else:
                scenes = torch.stack([b[0] for b in batch]).to(DEVICE)
                logits = model(scenes, tokens)

            target = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            if is_dual:
                train_acc = evaluate_exact_match_s5(model, train_data[:40])
                held_acc = evaluate_exact_match_s5(model, held_data)
            else:
                train_acc = evaluate_exact_match_s3(model, train_data[:60])
                held_acc = evaluate_exact_match_s3(model, held_data)
            print(f"  [{label}] epoch {epoch:3d} | train_acc={train_acc:.3f} | held_out_acc={held_acc:.3f}")

    if is_dual:
        final_held = evaluate_exact_match_s5(model, held_data)
    else:
        final_held = evaluate_exact_match_s3(model, held_data)
    return final_held


def run_all_baselines(epochs=EPOCHS_STAGE7_BASELINE):
    results = {}

    print("=" * 70)
    print("BASELINE — GRU Seq2Seq (no attention, no factored encoding)")
    print("=" * 70)

    def build_s3_data_baseline(pairs, n_per):
        data = []
        for (obj1, rel, obj2) in pairs:
            scene_vec = encode_scene(obj1, rel, obj2)
            tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
            text = sentence_text_canonical(obj1, rel, obj2)
            for _ in range(n_per):
                data.append((scene_vec, tokens, attn_targets, text))
        random.shuffle(data)
        return data

    print("\n[Baseline] Stage 3-equivalent test (plain sentences)...")
    bl3_train = build_s3_data_baseline(train_pairs, N_PER_TRAIN_COMBO)
    bl3_held = build_s3_data_baseline(held_pairs, N_PER_HELDOUT_COMBO)
    model_bl3 = GRUBaselineModel(SCENE_DIM).to(DEVICE)
    results["stage3"] = _train_baseline_generic(
        model_bl3, bl3_train, bl3_held, epochs, is_dual=False, label="S3"
    )

    print("\n[Baseline] Stage 5-equivalent test (paragraphs)...")
    def build_s5_data_baseline(fact_pool, n_paragraphs):
        data = []
        for _ in range(n_paragraphs):
            fact1 = random.choice(fact_pool)
            fact2 = random.choice(fact_pool)
            sv1 = encode_scene(*fact1)
            sv2 = encode_scene(*fact2)
            tokens, attn_targets = tokenize_paragraph_with_attn(fact1, fact2)
            text = paragraph_text(fact1, fact2)
            data.append((sv1, sv2, tokens, attn_targets, text))
        return data

    bl5_train = build_s5_data_baseline(train_pairs, N_TRAIN_PARAGRAPHS)
    bl5_held = build_s5_data_baseline(held_pairs, N_HELD_PARAGRAPHS)
    model_bl5 = GRUBaselineModelDual(SCENE_DIM).to(DEVICE)
    results["stage5"] = _train_baseline_generic(
        model_bl5, bl5_train, bl5_held, epochs, is_dual=True, label="S5"
    )

    print("\n[Baseline] Stage 6-equivalent test (compositionality)...")
    comp_train_pairs, comp_test_pairs, held_rel, held_shape = build_compositionality_split()
    bl6_train = build_s3_data_baseline(comp_train_pairs, 3)
    bl6_held = build_s3_data_baseline(comp_test_pairs, 2)
    model_bl6 = GRUBaselineModel(SCENE_DIM).to(DEVICE)
    results["stage6"] = _train_baseline_generic(
        model_bl6, bl6_train, bl6_held, epochs, is_dual=False, label="S6"
    )

    print("\n" + "=" * 70)
    print("GRU baseline results (compare with the Full-model numbers printed")
    print("for Stage 3/5/6 earlier in the same run)")
    print("=" * 70)
    print(f"  {'Test':10s} | {'GRU Baseline':13s}")
    print("  " + "-" * 28)
    print(f"  {'Stage3':10s} | {results['stage3']:.3f}")
    print(f"  {'Stage5':10s} | {results['stage5']:.3f}")
    print(f"  {'Stage6':10s} | {results['stage6']:.3f}")
    print()

    return results


# ═══════════════════════════════════════════════════════════════
# STAGE 8 — Multi-Split Compositional Benchmark
# ═══════════════════════════════════════════════════════════════
def build_compositionality_split_param(held_relation, held_shape_as_obj1):
    comp_train_pairs, comp_test_pairs = [], []
    all_objs_comp = all_objects()

    attempts = 0
    while len(comp_train_pairs) < COMPOSITIONALITY_N_TRAIN_PAIRS and attempts < 5000:
        attempts += 1
        o1 = random.choice(all_objs_comp)
        o2 = random.choice(all_objs_comp)
        if o1 == o2:
            continue
        rel = random.choice(RELATIONS)
        if rel == held_relation and o1[0] == held_shape_as_obj1:
            continue
        comp_train_pairs.append((o1, rel, o2))

    attempts = 0
    while len(comp_test_pairs) < COMPOSITIONALITY_N_TEST_PAIRS and attempts < 3000:
        attempts += 1
        o1 = random.choice(all_objs_comp)
        o2 = random.choice(all_objs_comp)
        if o1 == o2:
            continue
        if o1[0] != held_shape_as_obj1:
            continue
        comp_test_pairs.append((o1, held_relation, o2))

    return comp_train_pairs, comp_test_pairs


def train_one_compositional_split(held_relation, held_shape, epochs=EPOCHS_STAGE8_MULTISPLIT):
    comp_train_pairs, comp_test_pairs = build_compositionality_split_param(
        held_relation, held_shape
    )
    if len(comp_test_pairs) < 5:
        print(f"  Skipping {held_relation}+{held_shape}: insufficient test data")
        return None

    model = Stage3Model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def build_data(pairs, n_per):
        data = []
        for (obj1, rel, obj2) in pairs:
            scene_vec = encode_scene(obj1, rel, obj2)
            tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
            text = sentence_text_canonical(obj1, rel, obj2)
            for _ in range(n_per):
                data.append((scene_vec, tokens, attn_targets, text))
        random.shuffle(data)
        return data

    train_data_local = build_data(comp_train_pairs, 3)
    test_data = build_data(comp_test_pairs, 2)

    for epoch in range(epochs):
        random.shuffle(train_data_local)
        use_aux = epoch >= 60
        for i in range(0, len(train_data_local), BATCH_SIZE):
            batch = train_data_local[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)
            tokens = pad_batch([b[1] for b in batch]).to(DEVICE)
            batch_attn_targets = [b[2] for b in batch]

            logits, raw_slots, attn = model(scenes, tokens, return_attn=True)
            target = tokens[:, 1:]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )
            attn_terms = []
            for b_idx, targets_list in enumerate(batch_attn_targets):
                for (pos, slot) in targets_list:
                    if pos < attn.size(1):
                        probs = attn[b_idx, pos, :].unsqueeze(0)
                        attn_terms.append(F.nll_loss(
                            torch.log(probs + 1e-8),
                            torch.tensor([slot], device=DEVICE)
                        ))
            attn_sup_loss = torch.stack(attn_terms).mean() if attn_terms else torch.tensor(0.0, device=DEVICE)

            if use_aux:
                aux_loss = model.slot_encoder.auxiliary_loss(scenes, raw_slots)
                loss = gen_loss + 0.15 * aux_loss + 0.5 * attn_sup_loss
            else:
                loss = gen_loss + 0.5 * attn_sup_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    final_acc = evaluate_exact_match_s3(model, test_data)
    status = "OK" if final_acc > 0.7 else ("PARTIAL" if final_acc > 0.4 else "FAIL")
    print(f"  {held_relation:10s} + {held_shape:10s} -> accuracy = {final_acc:.3f}  [{status}]")
    return final_acc


def run_multi_split_benchmark(epochs=EPOCHS_STAGE8_MULTISPLIT):
    print("=" * 70)
    print("STAGE 8 — MULTI-SPLIT COMPOSITIONAL BENCHMARK")
    print("=" * 70)
    print("   Each split withholds a different (relation, shape) combination")
    print("   from training; accuracy is measured independently per split.\n")

    splits_to_test = [
        ("above", "circle"),
        ("above", "square"),
        ("below", "star"),
        ("below", "triangle"),
        ("left of", "circle"),
        ("left of", "star"),
        ("right of", "square"),
        ("right of", "triangle"),
    ]

    results = {}
    for held_rel, held_shape in splits_to_test:
        acc = train_one_compositional_split(held_rel, held_shape, epochs=epochs)
        if acc is not None:
            results[f"{held_rel}+{held_shape}"] = acc

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  {'Split':25s} | {'Accuracy':10s}")
    print("  " + "-" * 40)
    for split_name, acc in results.items():
        status = "OK" if acc > 0.7 else ("PARTIAL" if acc > 0.4 else "FAIL")
        print(f"  {split_name:25s} | {acc:.3f}      [{status}]")

    values = list(results.values())
    mean_acc = float(np.mean(values)) if values else 0.0
    std_acc = float(np.std(values)) if values else 0.0
    print("  " + "-" * 40)
    print(f"  {'Mean +/- Std':25s} | {mean_acc:.3f} +/- {std_acc:.3f}")

    print()
    if mean_acc > 0.8:
        print("Strong evidence that compositionality is a general behavior, not a single lucky split.")
    elif mean_acc > 0.5:
        print("Reasonable evidence, with variance — some combinations are harder than others.")
    else:
        print("The first split (above+circle) looks like an outlier rather than a general pattern.")

    return results, mean_acc, std_acc


# ═══════════════════════════════════════════════════════════════
# STAGE 9 — Ablation Study: isolating each component's contribution
# ═══════════════════════════════════════════════════════════════
"""
Design rationale.

Stage 7 compares the full architecture against a GRU baseline — an
"everything" vs. "nothing" comparison. The gap (e.g. compositional
accuracy 1.000 vs. 0.400, confirmed across 8 splits in Stage 8) is real,
but conflates four separate components:

  1) factored slot encoder (splitting the scene into separate shape/color/
     size fields instead of one merged representation)
  2) slot identity embeddings (solves attribute binding between the two
     objects)
  3) auxiliary disentanglement loss (regularizes each slot to keep clean,
     separated information)
  4) supervised attention alignment (directs attention to the correct
     slot at each output step, instead of letting it discover alignment
     unsupervised)

Without isolating them, any claim like "the factored encoder is the
reason" or "attention supervision is the reason" is unsupported. The
ablations below are truncated copies of Stage3Model, each missing exactly
one component, trained on identical data/epochs/optimizer settings so
that any difference in outcome is attributable to the removed component.

Each variant is evaluated on the same two tests: a plain held-out sentence
test (Stage 3-equivalent) and a true compositional split (Stage 6-
equivalent), since that is the strongest signal available.
"""

class AblationMLPEncoder(nn.Module):
    """
    Ablation A — removes factored encoding entirely.
    Instead of splitting the scene into (shape/color/size/relation), the
    whole scene vector goes through one shared MLP, followed by 7 separate
    linear heads that turn the same merged representation into 7 memory
    slots. There is no architectural signal separating "object 1's shape"
    from "object 1's color" — everything derives from the same undifferentiated
    representation.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SCENE_DIM, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.slot_heads = nn.ModuleList([nn.Linear(D, D) for _ in range(7)])

    def forward(self, scene_vec, slot_offset=0, return_raw=False, use_identity=True):
        h = self.net(scene_vec)                        # (B, D) merged representation
        slots = [head(h) for head in self.slot_heads]   # 7 projections from the same source
        memory = torch.stack(slots, dim=1)              # (B, 7, D)
        if return_raw:
            return memory, slots
        return memory

    def auxiliary_loss(self, scene_vec, raw_slots):
        # Not applicable here (no explicit attribute separation) — always zero
        return torch.tensor(0.0, device=scene_vec.device)


class AblationStage3Model(nn.Module):
    """
    Same architecture as Stage3Model (same AttentionDecoder, same training
    procedure). The only difference: it takes slot_encoder as a parameter
    so it can be swapped between experiments.
    """
    def __init__(self, slot_encoder):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = slot_encoder
        self.decoder = AttentionDecoder(self.embed)

    def forward(self, scene_vec, target_tokens, use_identity=True, return_attn=False):
        memory, raw_slots = self.slot_encoder(
            scene_vec, return_raw=True, use_identity=use_identity
        )
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw_slots, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw_slots

    @torch.no_grad()
    def describe(self, scene_vec, use_identity=True):
        self.eval()
        memory = self.slot_encoder(
            scene_vec.unsqueeze(0).to(DEVICE), use_identity=use_identity
        )
        result = self.decoder.generate(memory)[0]
        self.train()
        return result


ABLATION_MODES = ["full", "no_factored", "no_attn_sup", "no_identity"]


def _make_ablation_model(mode):
    """Builds the model and returns the training flags for the requested mode."""
    if mode == "no_factored":
        encoder = AblationMLPEncoder()
        use_aux = False   # not applicable without separated slots
    else:
        encoder = Stage3SlotEncoder()
        use_aux = True

    model = AblationStage3Model(encoder).to(DEVICE)
    use_attn_sup = (mode != "no_attn_sup")
    use_identity = (mode != "no_identity")
    return model, use_aux, use_attn_sup, use_identity


def _train_ablation_on_data(model, train_data, use_aux, use_attn_sup, use_identity,
                             epochs, label=""):
    """
    One training loop shared by all four modes — the only difference between
    them is which flags get passed (use_aux / use_attn_sup / use_identity),
    not the code itself. This ensures any difference in results is caused by
    the removed component, not by a different training loop.
    """
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        random.shuffle(train_data)
        warmup_ok = epoch >= 60
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i + BATCH_SIZE]
            scenes = torch.stack([b[0] for b in batch]).to(DEVICE)
            tokens = pad_batch([b[1] for b in batch]).to(DEVICE)
            batch_attn_targets = [b[2] for b in batch]

            logits, raw_slots, attn = model(
                scenes, tokens, use_identity=use_identity, return_attn=True
            )
            target = tokens[:, 1:]
            gen_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )

            loss = gen_loss

            if use_attn_sup:
                attn_terms = []
                for b_idx, targets_list in enumerate(batch_attn_targets):
                    for (pos, slot) in targets_list:
                        if pos < attn.size(1):
                            probs = attn[b_idx, pos, :].unsqueeze(0)
                            attn_terms.append(F.nll_loss(
                                torch.log(probs + 1e-8),
                                torch.tensor([slot], device=DEVICE)
                            ))
                attn_sup_loss = torch.stack(attn_terms).mean() if attn_terms else torch.tensor(0.0, device=DEVICE)
                loss = loss + 0.5 * attn_sup_loss
            # in no_attn_sup: attention is learned freely from gen_loss alone,
            # with no explicit supervision on which slot to attend to

            if use_aux and warmup_ok:
                aux_loss = model.slot_encoder.auxiliary_loss(scenes, raw_slots)
                loss = loss + 0.15 * aux_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    return model


@torch.no_grad()
def _evaluate_ablation(model, data, use_identity):
    model.eval()
    correct = 0
    for scene_vec, tokens, attn_targets, correct_text in data:
        generated = model.describe(scene_vec, use_identity=use_identity)
        if generated.strip() == correct_text.strip():
            correct += 1
    model.train()
    return correct / len(data) if len(data) > 0 else 0.0


def run_ablation_study(epochs=EPOCHS_STAGE9_ABLATION):
    """
    Trains the four variants (full + 3 ablations) on the same two tests:
    Stage 3-equivalent (plain sentences, held_pairs) and Stage 6-equivalent
    (a real compositional split, above+circle) — to see each component's
    contribution to both the ordinary and the compositional case together.
    """
    print("=" * 70)
    print("STAGE 9 — ABLATION STUDY (isolating each component's contribution)")
    print("=" * 70)

    def build_s3_data(pairs, n_per):
        data = []
        for (obj1, rel, obj2) in pairs:
            scene_vec = encode_scene(obj1, rel, obj2)
            tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
            text = sentence_text_canonical(obj1, rel, obj2)
            for _ in range(n_per):
                data.append((scene_vec, tokens, attn_targets, text))
        random.shuffle(data)
        return data

    # Test 1: same split as Stage 3 (ordinary held-out combinations)
    ab_s3_train = build_s3_data(train_pairs, N_PER_TRAIN_COMBO)
    ab_s3_held = build_s3_data(held_pairs, N_PER_HELDOUT_COMBO)

    # Test 2: same split as Stage 6 (a real compositional split — above+circle)
    comp_train_pairs, comp_test_pairs, held_rel, held_shape = build_compositionality_split()
    ab_comp_train = build_s3_data(comp_train_pairs, 3)
    ab_comp_test = build_s3_data(comp_test_pairs, 2)

    print(f"   Test 1 (Stage3-equiv): {len(ab_s3_train)} train | {len(ab_s3_held)} test")
    print(f"   Test 2 (Stage6-equiv, held='{held_rel}+{held_shape}'): "
          f"{len(ab_comp_train)} train | {len(ab_comp_test)} test\n")

    results = {}
    for mode in ABLATION_MODES:
        print(f"  Training variant: {mode} ...")
        model_s3, use_aux, use_attn_sup, use_identity = _make_ablation_model(mode)
        _train_ablation_on_data(
            model_s3, ab_s3_train, use_aux, use_attn_sup, use_identity, epochs, label=mode
        )
        acc_s3 = _evaluate_ablation(model_s3, ab_s3_held, use_identity)

        model_comp, use_aux2, use_attn_sup2, use_identity2 = _make_ablation_model(mode)
        _train_ablation_on_data(
            model_comp, ab_comp_train, use_aux2, use_attn_sup2, use_identity2, epochs, label=mode
        )
        acc_comp = _evaluate_ablation(model_comp, ab_comp_test, use_identity2)

        results[mode] = {"stage3_equiv": acc_s3, "stage6_equiv": acc_comp}
        print(f"    stage3_equiv_acc={acc_s3:.3f} | stage6_equiv_acc={acc_comp:.3f}\n")

    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"  {'Variant':16s} | {'Stage3-equiv':12s} | {'Stage6-equiv':12s}")
    print("  " + "-" * 46)
    for mode in ABLATION_MODES:
        r = results[mode]
        tag = "  <- FULL MODEL" if mode == "full" else ""
        print(f"  {mode:16s} | {r['stage3_equiv']:.3f}        | {r['stage6_equiv']:.3f}{tag}")
    print()

    full_comp = results["full"]["stage6_equiv"]
    print("Drop from FULL on the stage6_equiv test, by removed component:")
    for mode in ABLATION_MODES:
        if mode == "full":
            continue
        drop = full_comp - results[mode]["stage6_equiv"]
        print(f"    removing [{mode:14s}] -> drop = {drop:+.3f}")
    print()
    print("   The largest drop indicates the component most important for")
    print("   compositional generalization.")
    print("   Note: single run per variant — full statistical significance")
    print("   requires repeating across multiple random seeds (see Stage 10).")

    return results


# ═══════════════════════════════════════════════════════════════
# STAGE 10 — General-Purpose Multi-Seed Evaluation Harness
# ═══════════════════════════════════════════════════════════════
"""
Goal: build the harness once, generically, so that any experiment (Full,
Ablation, GRU, Transformer, ...) runs automatically across N seeds and
produces a mean +/- std table for Stage3 / Stage5 / Stage6, without
rewriting training/evaluation code for each new experiment.

Architecture: each experiment is an "ExperimentSpec" (dict) describing:
  - make_model()      : builds a fresh model instance
  - forward_fn(...)    : returns (logits, attn_or_None, raw_slots_or_None)
  - aux_loss_fn(...)   : computes auxiliary loss if the model supports it,
                         else None
  - describe_fn(...)   : generates the sentence at evaluation time

Since Stage3/Stage6 are "single scene" and Stage5 is "two scenes", every
experiment has a "single" and a "dual" spec. The generic training/eval
loops (`_generic_train_single/dual`, `_generic_eval_single/dual`) run
identically for any spec — the difference between full/ablation/GRU/
Transformer lives entirely in the spec, not in the training loop.

Note: variance across seeds here covers more than weight initialization —
train/held splits are rebuilt from scratch per seed (build_world_split /
build_comp_split), so the reported variance captures split randomness and
training randomness together, not just weight init.
"""

# ─────────────────────────────────────────────────────
# 10.1 — Rebuilding the world (splits) for any seed
# ─────────────────────────────────────────────────────
def build_world_split(seed):
    """
    Same logic as the train_pairs/held_pairs construction above, exposed as
    a reusable function for any seed. Uses fully local variables so it
    never touches the global train_pairs/held_pairs that Stages 1-9 depend on.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    local_all_objs = all_objects()
    random.shuffle(local_all_objs)

    seen_s, seen_c, seen_z = set(), set(), set()
    l_train_objs, l_extra_objs = [], []
    for obj in local_all_objs:
        s, c, z = obj
        if s not in seen_s or c not in seen_c or z not in seen_z:
            l_train_objs.append(obj)
            seen_s.add(s); seen_c.add(c); seen_z.add(z)
        else:
            l_extra_objs.append(obj)

    random.shuffle(l_extra_objs)
    l_train_objs = l_train_objs + l_extra_objs[:6]
    l_held_out_objs = l_extra_objs[6:14]

    l_exclude_set = set()
    for obj in l_train_objs + l_held_out_objs:
        for rel, shp in HELD_RELATION_SHAPE:
            if obj[0] == shp:
                l_exclude_set.add((rel, obj))

    l_train_pairs = make_pairs(l_train_objs, N_TRAIN_PAIRS_BASE, exclude_relation_obj=l_exclude_set)

    l_held_pairs_new_objects = make_pairs(l_held_out_objs, N_HELD_PAIRS_BASE)
    l_held_pairs_relation_shape = []
    attempts = 0
    while len(l_held_pairs_relation_shape) < N_HELD_PAIRS_BASE and attempts < 500:
        attempts += 1
        rel, shp = random.choice(list(HELD_RELATION_SHAPE))
        matching_objs = [o for o in l_train_objs + l_held_out_objs if o[0] == shp]
        if not matching_objs:
            continue
        o1 = random.choice(matching_objs)
        o2 = random.choice(l_train_objs)
        if o1 == o2:
            continue
        l_held_pairs_relation_shape.append((o1, rel, o2))

    l_held_pairs = l_held_pairs_new_objects + l_held_pairs_relation_shape
    return {"train_pairs": l_train_pairs, "held_pairs": l_held_pairs}


def build_comp_split(seed, held_relation="above", held_shape="circle"):
    """Same idea as build_world_split, for Stage 6's compositional split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return build_compositionality_split_param(held_relation, held_shape)


def build_single_scene_dataset(pairs, n_per):
    """Reusable module-level version of build_s3_data (duplicated locally in a few places)."""
    data = []
    for (obj1, rel, obj2) in pairs:
        scene_vec = encode_scene(obj1, rel, obj2)
        tokens, attn_targets = tokenize_canonical_with_attn(obj1, rel, obj2)
        text = sentence_text_canonical(obj1, rel, obj2)
        for _ in range(n_per):
            data.append((scene_vec, tokens, attn_targets, text))
    random.shuffle(data)
    return data


def build_dual_scene_dataset(fact_pool, n_paragraphs):
    """Reusable module-level version of build_s5_data (duplicated locally in a few places)."""
    data = []
    for _ in range(n_paragraphs):
        fact1 = random.choice(fact_pool)
        fact2 = random.choice(fact_pool)
        sv1 = encode_scene(*fact1)
        sv2 = encode_scene(*fact2)
        tokens, attn_targets = tokenize_paragraph_with_attn(fact1, fact2)
        text = paragraph_text(fact1, fact2)
        data.append((sv1, sv2, tokens, attn_targets, text))
    return data


# ─────────────────────────────────────────────────────
# 10.2 — Dual-scene ablation model (generalizes Stage5Model the
# same way AblationStage3Model does, so ablations are also
# available on Stage 5, not just Stage 3/6)
# ─────────────────────────────────────────────────────
class AblationStage5Model(nn.Module):
    """Same as Stage5Model, but slot_encoder and use_identity are swappable."""
    def __init__(self, slot_encoder):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = slot_encoder
        self.decoder = AttentionDecoder(self.embed)

    def encode_two_facts(self, scene_vec1, scene_vec2, use_identity=True):
        memory1, raw1 = self.slot_encoder(
            scene_vec1, slot_offset=0, return_raw=True, use_identity=use_identity
        )
        memory2, raw2 = self.slot_encoder(
            scene_vec2, slot_offset=7, return_raw=True, use_identity=use_identity
        )
        memory = torch.cat([memory1, memory2], dim=1)
        return memory, raw1, raw2

    def forward(self, scene_vec1, scene_vec2, target_tokens, use_identity=True, return_attn=False):
        memory, raw1, raw2 = self.encode_two_facts(scene_vec1, scene_vec2, use_identity=use_identity)
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw1, raw2, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw1, raw2

    @torch.no_grad()
    def describe(self, scene_vec1, scene_vec2, use_identity=True):
        self.eval()
        memory, _, _ = self.encode_two_facts(
            scene_vec1.unsqueeze(0).to(DEVICE),
            scene_vec2.unsqueeze(0).to(DEVICE),
            use_identity=use_identity,
        )
        result = self.decoder.generate(memory, max_len=26)[0]
        self.train()
        return result


# ─────────────────────────────────────────────────────
# 10.3 — Experiment specs (uniform adapters per model type)
# ─────────────────────────────────────────────────────
def _forward_full_single(model, scenes, tokens):
    logits, raw_slots, attn = model(scenes, tokens, return_attn=True)
    return logits, attn, raw_slots

def _aux_full_single(model, scenes, raw_slots):
    return model.slot_encoder.auxiliary_loss(scenes, raw_slots)

def _describe_full_single(model, scene_vec):
    return model.describe(scene_vec)

def _forward_full_dual(model, sv1, sv2, tokens):
    logits, raw1, raw2, attn = model(sv1, sv2, tokens, return_attn=True)
    return logits, attn, (raw1, raw2)

def _aux_full_dual(model, sv1, sv2, raw_pair):
    raw1, raw2 = raw_pair
    return model.slot_encoder.auxiliary_loss(sv1, raw1) + model.slot_encoder.auxiliary_loss(sv2, raw2)

def _describe_full_dual(model, sv1, sv2):
    return model.describe(sv1, sv2)


def _make_ablation_single_spec(name, encoder_factory, use_attn_sup, use_identity, has_aux):
    def make_model():
        return AblationStage3Model(encoder_factory())

    def forward_fn(model, scenes, tokens, _id=use_identity, _sup=use_attn_sup, _aux=has_aux):
        logits, raw_slots, attn = model(scenes, tokens, use_identity=_id, return_attn=True)
        return logits, (attn if _sup else None), (raw_slots if _aux else None)

    def describe_fn(model, scene_vec, _id=use_identity):
        return model.describe(scene_vec, use_identity=_id)

    return {
        "name": name, "kind": "single",
        "make_model": make_model, "forward_fn": forward_fn,
        "aux_loss_fn": (lambda model, scenes, raw: model.slot_encoder.auxiliary_loss(scenes, raw)) if has_aux else None,
        "describe_fn": describe_fn,
    }


def _make_ablation_dual_spec(name, encoder_factory, use_attn_sup, use_identity, has_aux):
    def make_model():
        return AblationStage5Model(encoder_factory())

    def forward_fn(model, sv1, sv2, tokens, _id=use_identity, _sup=use_attn_sup, _aux=has_aux):
        logits, raw1, raw2, attn = model(sv1, sv2, tokens, use_identity=_id, return_attn=True)
        return logits, (attn if _sup else None), ((raw1, raw2) if _aux else None)

    def aux_fn(model, sv1, sv2, raw_pair):
        raw1, raw2 = raw_pair
        return model.slot_encoder.auxiliary_loss(sv1, raw1) + model.slot_encoder.auxiliary_loss(sv2, raw2)

    def describe_fn(model, sv1, sv2, _id=use_identity):
        return model.describe(sv1, sv2, use_identity=_id)

    return {
        "name": name, "kind": "dual",
        "make_model": make_model, "forward_fn": forward_fn,
        "aux_loss_fn": aux_fn if has_aux else None,
        "describe_fn": describe_fn,
    }


def get_experiment_registry():
    """
    Single extension point: any new experiment (a Transformer, say) is added
    here in two lines (single + dual spec), and it runs automatically with
    run_multi_seed_benchmark and the rest of the harness with no other code
    changes required.
    """
    registry = {
        "Full": {
            "single": {
                "name": "Full", "kind": "single",
                "make_model": lambda: Stage3Model(),
                "forward_fn": _forward_full_single,
                "aux_loss_fn": _aux_full_single,
                "describe_fn": _describe_full_single,
            },
            "dual": {
                "name": "Full", "kind": "dual",
                "make_model": lambda: Stage5Model(),
                "forward_fn": _forward_full_dual,
                "aux_loss_fn": _aux_full_dual,
                "describe_fn": _describe_full_dual,
            },
        },
        "No Factored": {
            "single": _make_ablation_single_spec(
                "No Factored", AblationMLPEncoder, use_attn_sup=True, use_identity=True, has_aux=False
            ),
            "dual": _make_ablation_dual_spec(
                "No Factored", AblationMLPEncoder, use_attn_sup=True, use_identity=True, has_aux=False
            ),
        },
        "No Attention": {
            "single": _make_ablation_single_spec(
                "No Attention", Stage3SlotEncoder, use_attn_sup=False, use_identity=True, has_aux=True
            ),
            "dual": _make_ablation_dual_spec(
                "No Attention", Stage3SlotEncoder, use_attn_sup=False, use_identity=True, has_aux=True
            ),
        },
        "No Identity": {
            "single": _make_ablation_single_spec(
                "No Identity", Stage3SlotEncoder, use_attn_sup=True, use_identity=False, has_aux=True
            ),
            "dual": _make_ablation_dual_spec(
                "No Identity", Stage3SlotEncoder, use_attn_sup=True, use_identity=False, has_aux=True
            ),
        },
        "GRU": {
            "single": {
                "name": "GRU", "kind": "single",
                "make_model": lambda: GRUBaselineModel(SCENE_DIM),
                "forward_fn": lambda model, scenes, tokens: (model(scenes, tokens), None, None),
                "aux_loss_fn": None,
                "describe_fn": lambda model, scene_vec: model.describe(scene_vec),
            },
            "dual": {
                "name": "GRU", "kind": "dual",
                "make_model": lambda: GRUBaselineModelDual(SCENE_DIM),
                "forward_fn": lambda model, sv1, sv2, tokens: (model(sv1, sv2, tokens), None, None),
                "aux_loss_fn": None,
                "describe_fn": lambda model, sv1, sv2: model.describe(sv1, sv2),
            },
        },
        # "Transformer": {"single": ..., "dual": ...}  <- see Stage 11 for the registered entries
    }
    return registry


# ─────────────────────────────────────────────────────
# 10.4 — Generic train/eval loops (same code for every experiment)
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# 10.3.5 — Performance: GPU-resident data + vectorized attention loss
# ─────────────────────────────────────────────────────
"""
The real bottleneck in the earlier version: a Python loop calling
F.nll_loss once per (sample, token) pair — roughly 16 samples x 7
positions = ~112 separate GPU calls per batch, per training step, per
epoch. Each small call incurs its own kernel-launch dispatch overhead
(independent of the actual compute), which is what starves the GPU and
shows up as low utilization despite a small model.

The fix has two parts:
  1) Build the full dataset as GPU tensors once before the epoch loop
     (instead of re-stacking/padding/moving data every batch).
  2) Compute the attention supervision loss as a single vectorized
     operation (gather + nll_loss) instead of a Python loop — identical
     math, without the hundreds of small kernel launches.
"""

def build_attn_supervision_tensors(attn_targets_batch, max_targets=None):
    """Converts a list of [(pos, slot), ...] per sample into padded tensors."""
    if max_targets is None:
        max_targets = max((len(t) for t in attn_targets_batch), default=1)
    B = len(attn_targets_batch)
    pos_idx = torch.zeros(B, max_targets, dtype=torch.long)
    slot_idx = torch.full((B, max_targets), -100, dtype=torch.long)   # -100 = standard ignore_index
    for b, targets_list in enumerate(attn_targets_batch):
        for k, (pos, slot) in enumerate(targets_list):
            pos_idx[b, k] = pos
            slot_idx[b, k] = slot
    return pos_idx, slot_idx


def vectorized_attention_loss(attn, pos_idx, slot_idx):
    """
    Vectorized replacement for the old Python-loop version — identical math,
    as a single GPU operation instead of hundreds of small calls.
    attn: (B, T-1, S) | pos_idx, slot_idx: (B, K)
    """
    B, T, S = attn.shape
    pos_idx = pos_idx.clamp(max=T - 1)
    gathered = attn.gather(1, pos_idx.unsqueeze(-1).expand(-1, -1, S))   # (B, K, S)
    log_probs = torch.log(gathered + 1e-8).reshape(-1, S)
    targets = slot_idx.reshape(-1)
    return F.nll_loss(log_probs, targets, ignore_index=-100)


def prepare_gpu_dataset_single(data):
    """Converts a full single-scene dataset to GPU tensors once, before training."""
    scenes = torch.stack([d[0] for d in data]).to(DEVICE)
    tokens = pad_batch([d[1] for d in data]).to(DEVICE)
    max_targets = max((len(d[2]) for d in data), default=1)
    pos_idx, slot_idx = build_attn_supervision_tensors([d[2] for d in data], max_targets)
    return {
        "scenes": scenes, "tokens": tokens,
        "pos_idx": pos_idx.to(DEVICE), "slot_idx": slot_idx.to(DEVICE),
    }


def prepare_gpu_dataset_dual(data):
    """Same idea for the paragraph (two-scene) dataset."""
    sv1 = torch.stack([d[0] for d in data]).to(DEVICE)
    sv2 = torch.stack([d[1] for d in data]).to(DEVICE)
    tokens = pad_batch([d[2] for d in data]).to(DEVICE)
    max_targets = max((len(d[3]) for d in data), default=1)
    pos_idx, slot_idx = build_attn_supervision_tensors([d[3] for d in data], max_targets)
    return {
        "sv1": sv1, "sv2": sv2, "tokens": tokens,
        "pos_idx": pos_idx.to(DEVICE), "slot_idx": slot_idx.to(DEVICE),
    }


def _generic_train_single(spec, train_data, epochs, warmup_epoch=60,
                           attn_sup_weight=0.5, aux_weight=0.15, batch_size=None):
    """
    Same math as the earlier per-experiment loops — the difference is purely
    performance: data is GPU-resident (converted once), batching uses
    torch.randperm + fancy indexing (all on GPU, no per-batch CPU<->GPU
    transfer), and the attention loss is vectorized. batch_size is optional
    (defaults to BATCH_SIZE); increase it if the dataset is small enough to
    use larger or even full-batch steps.
    """
    model = spec["make_model"]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bs = batch_size or BATCH_SIZE

    ds = prepare_gpu_dataset_single(train_data)
    scenes_all, tokens_all = ds["scenes"], ds["tokens"]
    pos_idx_all, slot_idx_all = ds["pos_idx"], ds["slot_idx"]
    N = scenes_all.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        warmup_ok = epoch >= warmup_epoch
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            scenes = scenes_all[idx]
            tokens = tokens_all[idx]
            pos_idx = pos_idx_all[idx]
            slot_idx = slot_idx_all[idx]

            logits, attn, raw_slots = spec["forward_fn"](model, scenes, tokens)
            target = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )

            if attn is not None:
                loss = loss + attn_sup_weight * vectorized_attention_loss(attn, pos_idx, slot_idx)

            if raw_slots is not None and spec["aux_loss_fn"] is not None and warmup_ok:
                aux = spec["aux_loss_fn"](model, scenes, raw_slots)
                if aux is not None:
                    loss = loss + aux_weight * aux

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    return model


def _generic_train_dual(spec, train_data, epochs, warmup_epoch=60,
                         attn_sup_weight=0.5, aux_weight=0.15, batch_size=None):
    """Same performance improvements as _generic_train_single — two-scene (Stage 5) version."""
    model = spec["make_model"]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bs = batch_size or BATCH_SIZE

    ds = prepare_gpu_dataset_dual(train_data)
    sv1_all, sv2_all, tokens_all = ds["sv1"], ds["sv2"], ds["tokens"]
    pos_idx_all, slot_idx_all = ds["pos_idx"], ds["slot_idx"]
    N = sv1_all.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        warmup_ok = epoch >= warmup_epoch
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            sv1 = sv1_all[idx]
            sv2 = sv2_all[idx]
            tokens = tokens_all[idx]
            pos_idx = pos_idx_all[idx]
            slot_idx = slot_idx_all[idx]

            logits, attn, raw_slots_pair = spec["forward_fn"](model, sv1, sv2, tokens)
            target = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), target.reshape(-1), ignore_index=PAD
            )

            if attn is not None:
                loss = loss + attn_sup_weight * vectorized_attention_loss(attn, pos_idx, slot_idx)

            if raw_slots_pair is not None and spec["aux_loss_fn"] is not None and warmup_ok:
                aux = spec["aux_loss_fn"](model, sv1, sv2, raw_slots_pair)
                if aux is not None:
                    loss = loss + aux_weight * aux

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    return model


@torch.no_grad()
def _generic_eval_single(spec, model, data):
    correct = 0
    for scene_vec, tokens, attn_targets, text in data:
        gen = spec["describe_fn"](model, scene_vec)
        if gen.strip() == text.strip():
            correct += 1
    return correct / len(data) if data else 0.0


@torch.no_grad()
def _generic_eval_dual(spec, model, data):
    correct = 0
    for sv1, sv2, tokens, attn_targets, text in data:
        gen = spec["describe_fn"](model, sv1, sv2)
        if gen.strip() == text.strip():
            correct += 1
    return correct / len(data) if data else 0.0


# ─────────────────────────────────────────────────────
# 10.5 — Main runner: multi-seed x every experiment x 3 tests
# ─────────────────────────────────────────────────────
def run_multi_seed_benchmark(
    experiment_names=None,
    seeds=MULTI_SEED_LIST,
    epochs_stage3=EPOCHS_STAGE10_SEED_S3,
    epochs_stage5=EPOCHS_STAGE10_SEED_S5,
    epochs_stage6=EPOCHS_STAGE10_SEED_S6,
    held_relation="above",
    held_shape="circle",
):
    """
    Compute cost note: each seed trains (number of experiments x 3 tests)
    independent models from scratch. With 5 experiments and 5 seeds that's
    75 full training runs. If time is limited, reduce seeds or epochs here
    first (or pass experiment_names to run a subset), and run the full
    version (5-10 seeds, full epochs) separately for publishable numbers.
    """
    registry = get_experiment_registry()
    if experiment_names is None:
        experiment_names = list(registry.keys())

    results = {name: {"stage3": [], "stage5": [], "stage6": []} for name in experiment_names}

    print("=" * 70)
    print(f"STAGE 10 — MULTI-SEED EVALUATION ({len(seeds)} seeds: {list(seeds)})")
    print("=" * 70)
    print(f"   Experiments: {experiment_names}")
    print(f"   epochs: stage3={epochs_stage3} | stage5={epochs_stage5} | stage6={epochs_stage6}\n")

    for seed in seeds:
        print(f"-- SEED {seed} --------------------------------")

        world = build_world_split(seed)
        tp, hp = world["train_pairs"], world["held_pairs"]

        s3_train_data = build_single_scene_dataset(tp, N_PER_TRAIN_COMBO)
        s3_held_data = build_single_scene_dataset(hp, N_PER_HELDOUT_COMBO)

        s5_train_data = build_dual_scene_dataset(tp, N_TRAIN_PARAGRAPHS)
        s5_held_data = build_dual_scene_dataset(hp, N_HELD_PARAGRAPHS)

        comp_train_pairs, comp_test_pairs = build_comp_split(seed, held_relation, held_shape)
        s6_train_data = build_single_scene_dataset(comp_train_pairs, 3)
        s6_test_data = build_single_scene_dataset(comp_test_pairs, 2)

        for exp_name in experiment_names:
            spec_pair = registry[exp_name]

            random.seed(seed * 1000 + 1); torch.manual_seed(seed * 1000 + 1)
            model_s3 = _generic_train_single(spec_pair["single"], s3_train_data, epochs_stage3)
            acc_s3 = _generic_eval_single(spec_pair["single"], model_s3, s3_held_data)

            random.seed(seed * 1000 + 2); torch.manual_seed(seed * 1000 + 2)
            model_s5 = _generic_train_dual(spec_pair["dual"], s5_train_data, epochs_stage5)
            acc_s5 = _generic_eval_dual(spec_pair["dual"], model_s5, s5_held_data)

            random.seed(seed * 1000 + 3); torch.manual_seed(seed * 1000 + 3)
            model_s6 = _generic_train_single(spec_pair["single"], s6_train_data, epochs_stage6)
            acc_s6 = _generic_eval_single(spec_pair["single"], model_s6, s6_test_data)

            results[exp_name]["stage3"].append(acc_s3)
            results[exp_name]["stage5"].append(acc_s5)
            results[exp_name]["stage6"].append(acc_s6)

            print(f"  [{exp_name:12s}] stage3={acc_s3:.3f} | stage5={acc_s5:.3f} | stage6={acc_s6:.3f}")
        print()

    print("=" * 70)
    print("MULTI-SEED BENCHMARK — FINAL TABLE (mean +/- std)")
    print("=" * 70)
    header = f"  {'Model':22s} | {'Stage3':16s} | {'Stage5':16s} | {'Stage6':16s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary = {}
    for exp_name in experiment_names:
        row, cells = {}, []
        for stage in ["stage3", "stage5", "stage6"]:
            vals = results[exp_name][stage]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 0.0
            row[stage] = (mean, std)
            cells.append(f"{mean:.3f} +/- {std:.3f}")
        summary[exp_name] = row
        print(f"  {exp_name:22s} | {cells[0]:16s} | {cells[1]:16s} | {cells[2]:16s}")
    print()

    return results, summary


# ═══════════════════════════════════════════════════════════════
# STAGE 11 — Transformer Baselines (Standard & Factored)
# ═══════════════════════════════════════════════════════════════
"""
The only question here: can a standard Transformer reach the same level of
compositional generalization as our model? The goal is not to build the
best possible Transformer or to graft our own ideas onto it.

Rules followed:
  1) Identical data — the same global train_pairs/held_pairs (same split
     as Stages 1-8), same build_compositionality_split_param for Stage
     6/8, no changes to the data itself.
  2) Identical training — same Adam optimizer, same LR, same epoch count
     (400, matching the GRU baseline in Stage 7), same vocabulary and the
     same weight-tied output projection (a general technique used by every
     model in the project, including GRU, not specific to our approach).
  3) The only difference is the architecture:

     Standard Transformer (Transformer A):
       Token/field embeddings -> positional encoding (standard sinusoidal)
       -> Transformer encoder -> Transformer decoder -> linear -> vocab.
       One unavoidable technical detail: separate linear projections per
       field type (shape/color/size/relation) are required because their
       dimensions differ (4 vs 4 vs 2 vs 4) — this is a technical
       necessity, not factored encoding in the sense used elsewhere in
       this project (no identity embeddings, no auxiliary disentanglement
       loss, no supervised attention). Self-attention must discover any
       binding between the two objects on its own, relying only on the
       standard positional encoding — the standard mechanism any
       Transformer uses to resolve "which token belongs where", not an
       addition of ours.

     Factored Transformer (Transformer B — additional comparison):
       Uses our Stage3SlotEncoder as-is (identity embeddings + auxiliary
       disentanglement loss, since they are core to this project's
       definition of "factored encoding") but with a standard Transformer
       decoder instead of our LSTM+attention. No supervised attention loss
       here: the notion of "one alignment per output step" is designed for
       single-head attention, and forcing it onto multi-head attention
       would be a new architectural addition with no clear justification —
       so the Transformer decoder learns cross-attention unsupervised.

  4) Same tests: Stage3/Stage5/Stage6 (here) and Stage8 (in
     run_stage8_comparison), compared directly against GRU and our full model.
"""

import math


class SinusoidalPositionalEncoding(nn.Module):
    """Standard positional encoding (Vaswani et al.) — no project-specific customization."""
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, D)

    def forward(self, x):
        """x: (B, T, D) — adds positional encoding starting from position 0."""
        return x + self.pe[:, :x.size(1), :].to(x.device)


class GenericTransformerDecoder(nn.Module):
    """
    Standard Transformer decoder (nn.TransformerDecoder), shared between the
    Standard and Factored Transformer variants. The only difference between
    them is which encoder prepares the memory; the decoder itself is a
    single, fully standard module.
    """
    def __init__(self, embed_table, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embed = embed_table
        self.pos_enc = SinusoidalPositionalEncoding(D)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(T, device):
        # Bool mask (True = disallowed) instead of float -inf — avoids a
        # PyTorch warning about mixing mask types (tgt_mask and
        # tgt_key_padding_mask must match), with no behavioral difference.
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, memory, target_tokens):
        """Teacher forcing — target_tokens: (B, T) including SOS/EOS."""
        inp = target_tokens[:, :-1]
        emb = self.pos_enc(self.embed(inp))
        T = inp.size(1)
        causal_mask = self._causal_mask(T, inp.device)
        pad_mask = (inp == PAD)
        out = self.transformer_decoder(
            tgt=emb, memory=memory, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask,
        )
        logits = out @ self.embed.weight.T
        return logits

    @torch.no_grad()
    def generate(self, memory, max_len=14):
        B = memory.size(0)
        device = memory.device
        tokens = torch.full((B, 1), SOS, dtype=torch.long, device=device)
        done = [False] * B
        words_per_batch = [[] for _ in range(B)]
        for _ in range(max_len):
            emb = self.pos_enc(self.embed(tokens))
            T = tokens.size(1)
            causal_mask = self._causal_mask(T, device)
            out = self.transformer_decoder(tgt=emb, memory=memory, tgt_mask=causal_mask)
            logits = out[:, -1] @ self.embed.weight.T
            next_token = logits.argmax(-1)
            tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
            for b in range(B):
                if done[b]:
                    continue
                tid = next_token[b].item()
                if tid == EOS:
                    done[b] = True
                else:
                    words_per_batch[b].append(id2word.get(tid, "<unk>"))
            if all(done):
                break
        return [" ".join(w) for w in words_per_batch]


class StandardTransformerFieldEncoder(nn.Module):
    """
    Standard encoder — no identity embeddings, no auxiliary loss. Separate
    projections per field are a technical necessity (different dimensions),
    not "factored encoding" in this project's sense. Self-attention must
    discover any binding on its own, relying only on standard positional
    encoding (a different offset per object/fact).
    """
    def __init__(self, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.shape_proj = nn.Linear(N_SHAPES, D)
        self.color_proj = nn.Linear(N_COLORS, D)
        self.size_proj = nn.Linear(N_SIZES, D)
        self.rel_proj = nn.Linear(N_RELATIONS, D)
        self.pos_enc = SinusoidalPositionalEncoding(D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _fields_to_tokens(self, scene_vec):
        obj1_vec = scene_vec[..., :OBJ_DIM]
        rel_vec = scene_vec[..., OBJ_DIM:OBJ_DIM + N_RELATIONS]
        obj2_vec = scene_vec[..., OBJ_DIM + N_RELATIONS:]

        def obj_tokens(obj_vec):
            sh = obj_vec[..., :N_SHAPES]
            co = obj_vec[..., N_SHAPES:N_SHAPES + N_COLORS]
            sz = obj_vec[..., N_SHAPES + N_COLORS:]
            return self.size_proj(sz), self.color_proj(co), self.shape_proj(sh)

        z1, c1, s1 = obj_tokens(obj1_vec)
        r = self.rel_proj(rel_vec)
        z2, c2, s2 = obj_tokens(obj2_vec)
        return [z1, c1, s1, r, z2, c2, s2]

    def forward(self, scene_vec, position_offset=0):
        tokens = self._fields_to_tokens(scene_vec)
        seq = torch.stack(tokens, dim=1)   # (B, 7, D)
        seq = seq + self.pos_enc.pe[:, position_offset:position_offset + 7, :].to(seq.device)
        memory = self.transformer_encoder(seq)
        return memory


class StandardTransformerModel(nn.Module):
    """Transformer A — the actual baseline, single scene (Stage3/Stage6)."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.encoder = StandardTransformerFieldEncoder()
        self.decoder = GenericTransformerDecoder(self.embed)

    def forward(self, scene_vec, target_tokens, return_attn=False):
        memory = self.encoder(scene_vec, position_offset=0)
        logits = self.decoder(memory, target_tokens)
        return logits, None

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        memory = self.encoder(scene_vec.unsqueeze(0).to(DEVICE), position_offset=0)
        result = self.decoder.generate(memory)[0]
        self.train()
        return result


class StandardTransformerModelDual(nn.Module):
    """Transformer A — two-scene version (Stage5)."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.encoder = StandardTransformerFieldEncoder()
        self.decoder = GenericTransformerDecoder(self.embed)

    def forward(self, sv1, sv2, target_tokens, return_attn=False):
        mem1 = self.encoder(sv1, position_offset=0)
        mem2 = self.encoder(sv2, position_offset=7)
        memory = torch.cat([mem1, mem2], dim=1)
        logits = self.decoder(memory, target_tokens)
        return logits, None

    @torch.no_grad()
    def describe(self, sv1, sv2):
        self.eval()
        mem1 = self.encoder(sv1.unsqueeze(0).to(DEVICE), position_offset=0)
        mem2 = self.encoder(sv2.unsqueeze(0).to(DEVICE), position_offset=7)
        memory = torch.cat([mem1, mem2], dim=1)
        result = self.decoder.generate(memory, max_len=26)[0]
        self.train()
        return result


class FactoredTransformerModel(nn.Module):
    """Transformer B — our factored encoder + a standard Transformer decoder, single scene."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = Stage3SlotEncoder()
        self.decoder = GenericTransformerDecoder(self.embed)

    def forward(self, scene_vec, target_tokens, return_attn=False):
        memory, raw_slots = self.slot_encoder(scene_vec, return_raw=True)
        logits = self.decoder(memory, target_tokens)
        return logits, raw_slots

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        memory = self.slot_encoder(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(memory)[0]
        self.train()
        return result


class FactoredTransformerModelDual(nn.Module):
    """Transformer B — two-scene version (Stage5)."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=PAD)
        self.slot_encoder = Stage3SlotEncoder()
        self.decoder = GenericTransformerDecoder(self.embed)

    def encode_two_facts(self, sv1, sv2):
        memory1, raw1 = self.slot_encoder(sv1, slot_offset=0, return_raw=True)
        memory2, raw2 = self.slot_encoder(sv2, slot_offset=7, return_raw=True)
        memory = torch.cat([memory1, memory2], dim=1)
        return memory, raw1, raw2

    def forward(self, sv1, sv2, target_tokens, return_attn=False):
        memory, raw1, raw2 = self.encode_two_facts(sv1, sv2)
        logits = self.decoder(memory, target_tokens)
        return logits, raw1, raw2

    @torch.no_grad()
    def describe(self, sv1, sv2):
        self.eval()
        memory, _, _ = self.encode_two_facts(sv1.unsqueeze(0).to(DEVICE), sv2.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(memory, max_len=26)[0]
        self.train()
        return result


# ─────────────────────────────────────────────────────
# 11.1 — Spec adapters (registers both Transformer experiments in the
# shared registry so they run automatically with run_multi_seed_benchmark
# and run_stage8_comparison without extra code)
# ─────────────────────────────────────────────────────
def _forward_standard_transformer_single(model, scenes, tokens):
    logits, _ = model(scenes, tokens)
    return logits, None, None

def _forward_standard_transformer_dual(model, sv1, sv2, tokens):
    logits, _ = model(sv1, sv2, tokens)
    return logits, None, None

def _forward_factored_transformer_single(model, scenes, tokens):
    logits, raw_slots = model(scenes, tokens)
    return logits, None, raw_slots

def _aux_factored_transformer_single(model, scenes, raw_slots):
    return model.slot_encoder.auxiliary_loss(scenes, raw_slots)

def _forward_factored_transformer_dual(model, sv1, sv2, tokens):
    logits, raw1, raw2 = model(sv1, sv2, tokens)
    return logits, None, (raw1, raw2)

def _aux_factored_transformer_dual(model, sv1, sv2, raw_pair):
    raw1, raw2 = raw_pair
    return model.slot_encoder.auxiliary_loss(sv1, raw1) + model.slot_encoder.auxiliary_loss(sv2, raw2)


def register_transformer_experiments(registry):
    """Adds Standard/Factored Transformer to the same registry used by Stage 10."""
    registry["Standard Transformer"] = {
        "single": {
            "name": "Standard Transformer", "kind": "single",
            "make_model": lambda: StandardTransformerModel(),
            "forward_fn": _forward_standard_transformer_single,
            "aux_loss_fn": None,
            "describe_fn": lambda model, sv: model.describe(sv),
        },
        "dual": {
            "name": "Standard Transformer", "kind": "dual",
            "make_model": lambda: StandardTransformerModelDual(),
            "forward_fn": _forward_standard_transformer_dual,
            "aux_loss_fn": None,
            "describe_fn": lambda model, sv1, sv2: model.describe(sv1, sv2),
        },
    }
    registry["Factored Transformer"] = {
        "single": {
            "name": "Factored Transformer", "kind": "single",
            "make_model": lambda: FactoredTransformerModel(),
            "forward_fn": _forward_factored_transformer_single,
            "aux_loss_fn": _aux_factored_transformer_single,
            "describe_fn": lambda model, sv: model.describe(sv),
        },
        "dual": {
            "name": "Factored Transformer", "kind": "dual",
            "make_model": lambda: FactoredTransformerModelDual(),
            "forward_fn": _forward_factored_transformer_dual,
            "aux_loss_fn": _aux_factored_transformer_dual,
            "describe_fn": lambda model, sv1, sv2: model.describe(sv1, sv2),
        },
    }
    return registry


# Register the Transformer experiments in the base registry immediately —
# from here on, get_experiment_registry() returns them automatically, and
# Stage 10 (multi-seed) picks them up with no further changes there.
_original_get_experiment_registry = get_experiment_registry
def get_experiment_registry():
    return register_transformer_experiments(_original_get_experiment_registry())


# ─────────────────────────────────────────────────────
# 11.2 — Head-to-head on the same global Stage 3/5/6 data
# (mirrors run_all_baselines for direct comparability)
# ─────────────────────────────────────────────────────
def run_transformer_baselines(epochs=EPOCHS_STAGE11_TRANSFORMER):
    registry = get_experiment_registry()
    results = {}

    for exp_name in ["Standard Transformer", "Factored Transformer"]:
        print("=" * 70)
        print(f"{exp_name.upper()} BASELINE")
        print("=" * 70)

        spec_single = registry[exp_name]["single"]
        spec_dual = registry[exp_name]["dual"]

        print(f"\n[{exp_name}] Stage 3-equivalent test (plain sentences)...")
        s3_train = build_single_scene_dataset(train_pairs, N_PER_TRAIN_COMBO)
        s3_held = build_single_scene_dataset(held_pairs, N_PER_HELDOUT_COMBO)
        model_s3 = _generic_train_single(spec_single, s3_train, epochs)
        acc_s3 = _generic_eval_single(spec_single, model_s3, s3_held)
        print(f"  held_out_acc = {acc_s3:.3f}")

        print(f"\n[{exp_name}] Stage 5-equivalent test (paragraphs)...")
        s5_train = build_dual_scene_dataset(train_pairs, N_TRAIN_PARAGRAPHS)
        s5_held = build_dual_scene_dataset(held_pairs, N_HELD_PARAGRAPHS)
        model_s5 = _generic_train_dual(spec_dual, s5_train, epochs)
        acc_s5 = _generic_eval_dual(spec_dual, model_s5, s5_held)
        print(f"  held_out_acc = {acc_s5:.3f}")

        print(f"\n[{exp_name}] Stage 6-equivalent test (compositionality)...")
        comp_train_pairs, comp_test_pairs, _, _ = build_compositionality_split()
        s6_train = build_single_scene_dataset(comp_train_pairs, 3)
        s6_held = build_single_scene_dataset(comp_test_pairs, 2)
        model_s6 = _generic_train_single(spec_single, s6_train, epochs)
        acc_s6 = _generic_eval_single(spec_single, model_s6, s6_held)
        print(f"  held_out_acc = {acc_s6:.3f}\n")

        results[exp_name] = {"stage3": acc_s3, "stage5": acc_s5, "stage6": acc_s6}

    print("=" * 70)
    print("Comparison: GRU vs. Standard Transformer vs. Factored Transformer")
    print("(compare manually with the Full-model numbers from Stage 3/5/6 in the same run)")
    print("=" * 70)
    print(f"  {'Model':22s} | {'Stage3':8s} | {'Stage5':8s} | {'Stage6':8s}")
    print("  " + "-" * 54)
    for name, r in results.items():
        print(f"  {name:22s} | {r['stage3']:.3f}    | {r['stage5']:.3f}    | {r['stage6']:.3f}")
    print()

    return results


# ─────────────────────────────────────────────────────
# 11.3 — Generic Stage 8: any single-scene experiment across the 8 splits
# ─────────────────────────────────────────────────────
def run_multi_split_benchmark_generic(spec, splits_to_test=None, epochs=EPOCHS_STAGE8_MULTISPLIT, label=""):
    if splits_to_test is None:
        splits_to_test = [
            ("above", "circle"), ("above", "square"),
            ("below", "star"), ("below", "triangle"),
            ("left of", "circle"), ("left of", "star"),
            ("right of", "square"), ("right of", "triangle"),
        ]
    print("=" * 70)
    print(f"MULTI-SPLIT COMPOSITIONAL BENCHMARK — {label}")
    print("=" * 70)

    results = {}
    for held_rel, held_shape in splits_to_test:
        comp_train_pairs, comp_test_pairs = build_compositionality_split_param(held_rel, held_shape)
        if len(comp_test_pairs) < 5:
            print(f"  Skipping {held_rel}+{held_shape}: insufficient data")
            continue
        train_data_local = build_single_scene_dataset(comp_train_pairs, 3)
        test_data = build_single_scene_dataset(comp_test_pairs, 2)
        model = _generic_train_single(spec, train_data_local, epochs)
        acc = _generic_eval_single(spec, model, test_data)
        status = "OK" if acc > 0.7 else ("PARTIAL" if acc > 0.4 else "FAIL")
        print(f"  {held_rel:10s} + {held_shape:10s} -> accuracy = {acc:.3f}  [{status}]")
        results[f"{held_rel}+{held_shape}"] = acc

    values = list(results.values())
    mean_acc = float(np.mean(values)) if values else 0.0
    std_acc = float(np.std(values)) if values else 0.0
    print(f"\n  Mean +/- Std = {mean_acc:.3f} +/- {std_acc:.3f}\n")
    return results, mean_acc, std_acc


def run_stage8_comparison(experiment_names=None, splits_to_test=None, epochs=EPOCHS_STAGE8_MULTISPLIT):
    """
    Generalized version of Stage 8 — runs on any experiment registered in
    the registry (Full, GRU, Standard Transformer, Factored Transformer, or
    any ablation) and produces a single comparison table.
    """
    registry = get_experiment_registry()
    if experiment_names is None:
        experiment_names = ["Full", "GRU", "Standard Transformer", "Factored Transformer"]

    summary = {}
    for name in experiment_names:
        spec = registry[name]["single"]
        _, mean_acc, std_acc = run_multi_split_benchmark_generic(
            spec, splits_to_test=splits_to_test, epochs=epochs, label=name
        )
        summary[name] = (mean_acc, std_acc)

    print("=" * 70)
    print("STAGE 8 COMPARISON — MULTI-SPLIT COMPOSITIONAL BENCHMARK")
    print("=" * 70)
    print(f"  {'Model':22s} | {'Mean +/- Std':16s}")
    print("  " + "-" * 42)
    for name, (mean_acc, std_acc) in summary.items():
        print(f"  {name:22s} | {mean_acc:.3f} +/- {std_acc:.3f}")
    print()

    return summary


# ═══════════════════════════════════════════════════════════════
# STAGE 13 — World Scaling + Severe OOD
# ═══════════════════════════════════════════════════════════════
"""
A different question from Stage 6/8: not "does the model generalize to a
withheld combination?" but "does performance collapse as the world grows?"
Instead of 4 shapes x 4 colors x 2 sizes x 4 relations, this stage scales
to 20-30 of each, and tracks three things together: accuracy, training
time, and GPU memory usage.

Deliberate scope decision: sentences here keep the original grammar (two
objects + one relation, 7 slots per fact). Extending to longer sentences
("A above B and B left of C ...") would require a structural redesign of
the number of objects/slots, not just scaling Config, and is left as a
separate future extension, contingent on this experiment holding up first.

Important design decision: the original classes (Stage3Model,
GRUBaselineModel, etc.) read VOCAB_SIZE/PAD/SOS/EOS/id2word/N_SHAPES... as
globals fixed to the original world size (4x4x2x4). Scaling the world here
does not touch them at all — instead this section builds a fully
independent "Scaled" variant (ScaledSlotEncoder, ScaledAttentionDecoder,
ScaledStage3Model, ScaledGRUBaselineModel) that takes every dimension as a
parameter, so that:
  (1) there's no risk of breaking the validated Stage 1-12 results, and
  (2) this experiment can be rerun at any world size with no further code
      changes.

Severe OOD: instead of withholding a single (relation, shape) combination
as in Stage 6, this reserves a set of "rare" shapes entirely from
co-occurring with each other in training — each rare shape still appears
frequently with ordinary shapes and across all relations (this is not
zero-shot in the sense of a token never seen at all, which is technically
impossible since an embedding can't learn without gradient), but no
(rare shape, rare shape) pair ever appears in training. The test set
consists only of pairs of rare shapes together — e.g. "hexagon above
pentagon", where neither shape has ever been paired with another rare
shape during training.
"""

import time as _time


# ─────────────────────────────────────────────────────
# 13.1 — Word pools (with a synthetic fallback if more words are requested
# than the pool contains)
# ─────────────────────────────────────────────────────
SHAPE_POOL = [
    "circle", "square", "triangle", "star", "hexagon", "pentagon", "octagon",
    "rectangle", "oval", "diamond", "rhombus", "trapezoid", "crescent", "heart",
    "arrow", "cross", "parallelogram", "semicircle", "cylinder", "cube",
    "sphere", "cone", "pyramid", "prism", "spiral", "zigzag", "arch", "wedge",
    "teardrop", "kite",
]
COLOR_POOL = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "teal", "maroon", "navy",
    "olive", "gold", "silver", "beige", "turquoise", "lavender", "crimson",
    "indigo", "coral", "mint", "peach", "charcoal", "amber", "violet",
]
SIZE_POOL = [
    "tiny", "small", "little", "compact", "petite", "modest", "medium",
    "sizable", "large", "big", "huge", "giant", "massive", "enormous",
    "colossal", "gigantic", "miniature", "wee", "hefty", "substantial",
    "grand", "immense", "vast", "mammoth", "monstrous",
]
RELATION_POOL = [
    "above", "below", "left of", "right of", "inside", "outside", "behind",
    "in front of", "touching", "near", "far from", "overlapping",
    "intersecting", "surrounding", "adjacent to", "beside", "between",
    "under", "over", "against", "around", "beneath", "atop", "alongside",
    "facing",
]


def build_scaled_vocab(n_shapes, n_colors, n_sizes, n_relations):
    """Returns lists of the requested size — drawn from the real pool, with a synthetic fallback."""
    def take(pool, n, tag):
        if n <= len(pool):
            return pool[:n]
        extra = [f"{tag}_{i}" for i in range(len(pool), n)]
        return pool + extra

    shapes = take(SHAPE_POOL, n_shapes, "shape")
    colors = take(COLOR_POOL, n_colors, "color")
    sizes = take(SIZE_POOL, n_sizes, "size")
    relations = take(RELATION_POOL, n_relations, "rel")
    return shapes, colors, sizes, relations


def build_world_vocab_and_ids(shapes, colors, sizes, relations):
    """
    Keeps the exact same <pad>/<sos>/<eos> ordering as the original
    (first 3 words) — so PAD=0, SOS=1, EOS=2 always, letting us reuse
    pad_batch/build_attn_supervision_tensors unchanged.
    """
    special = ["<pad>", "<sos>", "<eos>"]
    function_words = {"the", "is", "."}
    for rel in relations:
        for w in rel.split():
            function_words.add(w)
    function_words = sorted(function_words)

    content = list(shapes) + list(colors) + list(sizes)
    words = special + function_words + content
    assert len(words) == len(set(words)), "duplicate word in the scaled vocabulary"
    word2id_local = {w: i for i, w in enumerate(words)}
    id2word_map = {i: w for w, i in word2id_local.items()}
    return {
        "words": words, "word2id": word2id_local, "id2word": id2word_map,
        "vocab_size": len(words),
        "pad": word2id_local["<pad>"], "sos": word2id_local["<sos>"], "eos": word2id_local["<eos>"],
    }


def build_scaled_world(n_shapes, n_colors, n_sizes, n_relations):
    """Builds a fully independent "world" (vocab + dimensions) that does not touch Stage 1-12 globals."""
    shapes, colors, sizes, relations = build_scaled_vocab(n_shapes, n_colors, n_sizes, n_relations)
    vocab = build_world_vocab_and_ids(shapes, colors, sizes, relations)
    n_s, n_c, n_z, n_r = len(shapes), len(colors), len(sizes), len(relations)
    obj_dim = n_s + n_c + n_z
    scene_dim = obj_dim * 2 + n_r
    world = {
        "shapes": shapes, "colors": colors, "sizes": sizes, "relations": relations,
        "n_shapes": n_s, "n_colors": n_c, "n_sizes": n_z, "n_relations": n_r,
        "obj_dim": obj_dim, "scene_dim": scene_dim,
        **vocab,
    }
    return world


# ─────────────────────────────────────────────────────
# 13.2 — Scene encoding / tokenization (generalized over world)
# ─────────────────────────────────────────────────────
def encode_object_scaled(shape, color, size, world):
    s = F.one_hot(torch.tensor(world["shapes"].index(shape)), world["n_shapes"]).float()
    c = F.one_hot(torch.tensor(world["colors"].index(color)), world["n_colors"]).float()
    z = F.one_hot(torch.tensor(world["sizes"].index(size)), world["n_sizes"]).float()
    return torch.cat([s, c, z])


def encode_scene_scaled(obj1, rel, obj2, world):
    o1 = encode_object_scaled(*obj1, world)
    o2 = encode_object_scaled(*obj2, world)
    r = F.one_hot(torch.tensor(world["relations"].index(rel)), world["n_relations"]).float()
    return torch.cat([o1, r, o2])


def tokenize_canonical_with_attn_scaled(obj1, rel, obj2, world):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2
    word2id_local = world["word2id"]
    words, attn_targets = [], []

    def add(w, slot=None):
        words.append(w)
        if slot is not None:
            attn_targets.append((len(words) - 1, slot))

    add("the")
    add(z1, 0); add(c1, 1); add(s1, 2)
    add("is")
    for w in rel.split():
        add(w, 3)
    add("the")
    add(z2, 4); add(c2, 5); add(s2, 6)
    add(".")

    tokens = [world["sos"]] + [word2id_local[w] for w in words] + [world["eos"]]
    return tokens, attn_targets


def sentence_text_canonical_scaled(obj1, rel, obj2):
    s1, c1, z1 = obj1
    s2, c2, z2 = obj2
    return f"the {z1} {c1} {s1} is {rel} the {z2} {c2} {s2} ."


def all_objects_scaled(world):
    return list(itertools.product(world["shapes"], world["colors"], world["sizes"]))


def build_scaled_dataset(pairs, world, n_per=2):
    data = []
    for (obj1, rel, obj2) in pairs:
        scene_vec = encode_scene_scaled(obj1, rel, obj2, world)
        tokens, attn_targets = tokenize_canonical_with_attn_scaled(obj1, rel, obj2, world)
        text = sentence_text_canonical_scaled(obj1, rel, obj2)
        for _ in range(n_per):
            data.append((scene_vec, tokens, attn_targets, text))
    random.shuffle(data)
    return data


# ─────────────────────────────────────────────────────
# 13.3 — Severe OOD split (shape pairs that never co-occurred at all)
# ─────────────────────────────────────────────────────
def build_scaled_split_with_severe_ood(world, seed, n_ood_shapes=None,
                                        n_train_pairs=500, n_test_pairs=60):
    # Default scales with world size instead of a fixed number: a fixed
    # default of 5 would break any world with fewer than 6 shapes. At
    # least one ordinary (non-OOD) shape must remain so training can still
    # build valid pairs.
    if n_ood_shapes is None:
        n_ood_shapes = max(1, min(5, world["n_shapes"] - 1))
    assert n_ood_shapes < world["n_shapes"], (
        f"n_ood_shapes ({n_ood_shapes}) must be less than n_shapes ({world['n_shapes']}) "
        f"— at least one ordinary shape must remain so training can build pairs"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    shapes = world["shapes"][:]
    random.shuffle(shapes)
    ood_shapes = set(shapes[:n_ood_shapes])

    all_objs_local = all_objects_scaled(world)

    def sample_object():
        return random.choice(all_objs_local)

    train_pairs_local = []
    attempts = 0
    while len(train_pairs_local) < n_train_pairs and attempts < n_train_pairs * 30:
        attempts += 1
        o1, o2 = sample_object(), sample_object()
        if o1 == o2:
            continue
        if o1[0] in ood_shapes and o2[0] in ood_shapes:
            continue   # a (rare, rare) pair is never allowed in training
        rel = random.choice(world["relations"])
        train_pairs_local.append((o1, rel, o2))

    train_pairs_set = set(train_pairs_local)

    normal_test_pairs = []
    attempts = 0
    while len(normal_test_pairs) < n_test_pairs and attempts < n_test_pairs * 30:
        attempts += 1
        o1, o2 = sample_object(), sample_object()
        if o1 == o2:
            continue
        if o1[0] in ood_shapes and o2[0] in ood_shapes:
            continue
        rel = random.choice(world["relations"])
        pair = (o1, rel, o2)
        if pair not in train_pairs_set:
            normal_test_pairs.append(pair)

    ood_test_pairs = []
    attempts = 0
    while len(ood_test_pairs) < n_test_pairs and attempts < n_test_pairs * 30:
        attempts += 1
        o1, o2 = sample_object(), sample_object()
        if o1 == o2:
            continue
        if not (o1[0] in ood_shapes and o2[0] in ood_shapes):
            continue
        rel = random.choice(world["relations"])
        ood_test_pairs.append((o1, rel, o2))

    return {
        "train_pairs": train_pairs_local,
        "normal_test_pairs": normal_test_pairs,
        "ood_test_pairs": ood_test_pairs,
        "ood_shapes": ood_shapes,
    }


# ─────────────────────────────────────────────────────
# 13.4 — Scaled model classes (dimensions as parameters, not globals)
# ─────────────────────────────────────────────────────
class ScaledSlotEncoder(nn.Module):
    """Generalized version of Stage3SlotEncoder — same logic, configurable dimensions."""
    def __init__(self, n_shapes, n_colors, n_sizes, n_relations, d=D):
        super().__init__()
        self.n_shapes, self.n_colors, self.n_sizes, self.n_relations = (
            n_shapes, n_colors, n_sizes, n_relations
        )
        self.shape_proj = nn.Linear(n_shapes, d)
        self.color_proj = nn.Linear(n_colors, d)
        self.size_proj = nn.Linear(n_sizes, d)
        self.rel_proj = nn.Linear(n_relations, d)
        self.slot_id_embed = nn.Embedding(14, d)   # 7 slots x two facts — fixed, independent of vocab size

        self.aux_shape = nn.Linear(d, n_shapes)
        self.aux_color = nn.Linear(d, n_colors)
        self.aux_size = nn.Linear(d, n_sizes)
        self.aux_rel = nn.Linear(d, n_relations)

    def forward(self, scene_vec, slot_offset=0, return_raw=False, use_identity=True):
        obj_dim = self.n_shapes + self.n_colors + self.n_sizes
        obj1_vec = scene_vec[..., :obj_dim]
        rel_vec = scene_vec[..., obj_dim:obj_dim + self.n_relations]
        obj2_vec = scene_vec[..., obj_dim + self.n_relations:]

        def obj_slots(obj_vec):
            sh = obj_vec[..., :self.n_shapes]
            co = obj_vec[..., self.n_shapes:self.n_shapes + self.n_colors]
            sz = obj_vec[..., self.n_shapes + self.n_colors:]
            return self.size_proj(sz), self.color_proj(co), self.shape_proj(sh)

        z1, c1, s1 = obj_slots(obj1_vec)
        r = self.rel_proj(rel_vec)
        z2, c2, s2 = obj_slots(obj2_vec)
        raw_slots = [z1, c1, s1, r, z2, c2, s2]

        if use_identity:
            slot_ids = torch.arange(slot_offset, slot_offset + 7, device=scene_vec.device)
            id_vecs = self.slot_id_embed(slot_ids)
            slots = [raw_slots[i] + id_vecs[i].unsqueeze(0) for i in range(7)]
        else:
            slots = raw_slots

        memory = torch.stack(slots, dim=1)
        if return_raw:
            return memory, raw_slots
        return memory

    def auxiliary_loss(self, scene_vec, raw_slots):
        z1, c1, s1, r, z2, c2, s2 = raw_slots
        obj_dim = self.n_shapes + self.n_colors + self.n_sizes
        obj1_vec = scene_vec[..., :obj_dim]
        rel_vec = scene_vec[..., obj_dim:obj_dim + self.n_relations]
        obj2_vec = scene_vec[..., obj_dim + self.n_relations:]

        shape1_t = obj1_vec[..., :self.n_shapes].argmax(-1)
        color1_t = obj1_vec[..., self.n_shapes:self.n_shapes + self.n_colors].argmax(-1)
        size1_t = obj1_vec[..., self.n_shapes + self.n_colors:].argmax(-1)
        rel_t = rel_vec.argmax(-1)
        shape2_t = obj2_vec[..., :self.n_shapes].argmax(-1)
        color2_t = obj2_vec[..., self.n_shapes:self.n_shapes + self.n_colors].argmax(-1)
        size2_t = obj2_vec[..., self.n_shapes + self.n_colors:].argmax(-1)

        loss = (
            F.cross_entropy(self.aux_size(z1), size1_t) +
            F.cross_entropy(self.aux_color(c1), color1_t) +
            F.cross_entropy(self.aux_shape(s1), shape1_t) +
            F.cross_entropy(self.aux_rel(r), rel_t) +
            F.cross_entropy(self.aux_size(z2), size2_t) +
            F.cross_entropy(self.aux_color(c2), color2_t) +
            F.cross_entropy(self.aux_shape(s2), shape2_t)
        )
        return loss


class ScaledAttentionDecoder(AttentionDecoder):
    """Same as AttentionDecoder, with SOS/EOS/id2word passed as parameters instead of globals."""
    def __init__(self, embed_table, sos_id, eos_id, id2word_map):
        super().__init__(embed_table)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.id2word_map = id2word_map

    @torch.no_grad()
    def generate(self, memory, max_len=14):
        B = memory.size(0)
        h = memory.mean(dim=1)
        c = torch.zeros_like(h)
        token = torch.full((B,), self.sos_id, dtype=torch.long, device=memory.device)
        words_per_batch = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            logits, h, c, _attn = self._step(token, h, c, memory)
            token = logits.argmax(-1)
            for b in range(B):
                if done[b]:
                    continue
                tid = token[b].item()
                if tid == self.eos_id:
                    done[b] = True
                else:
                    words_per_batch[b].append(self.id2word_map.get(tid, "<unk>"))
            if all(done):
                break
        return [" ".join(w) for w in words_per_batch]


class ScaledStage3Model(nn.Module):
    """Generalized version of Stage3Model (the full model) — works at any world size."""
    def __init__(self, world):
        super().__init__()
        self.embed = nn.Embedding(world["vocab_size"], D, padding_idx=world["pad"])
        self.slot_encoder = ScaledSlotEncoder(
            world["n_shapes"], world["n_colors"], world["n_sizes"], world["n_relations"]
        )
        self.decoder = ScaledAttentionDecoder(self.embed, world["sos"], world["eos"], world["id2word"])

    def forward(self, scene_vec, target_tokens, return_attn=False):
        memory, raw_slots = self.slot_encoder(scene_vec, return_raw=True)
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw_slots, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw_slots

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        memory = self.slot_encoder(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(memory)[0]
        self.train()
        return result


class ScaledGRUBaselineDecoder(GRUBaselineDecoder):
    """Same as GRUBaselineDecoder — SOS/EOS/id2word passed as parameters."""
    def __init__(self, embed_table, sos_id, eos_id, id2word_map):
        super().__init__(embed_table)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.id2word_map = id2word_map

    @torch.no_grad()
    def generate(self, scene_embed, max_len=14):
        B = scene_embed.size(0)
        h = scene_embed.unsqueeze(0)
        token = torch.full((B,), self.sos_id, dtype=torch.long, device=scene_embed.device)
        words_per_batch = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            emb = self.embed(token).unsqueeze(1)
            out, h = self.gru(emb, h)
            logits = self.out_proj(out[:, -1]) @ self.embed.weight.T
            token = logits.argmax(-1)
            for b in range(B):
                if done[b]:
                    continue
                tid = token[b].item()
                if tid == self.eos_id:
                    done[b] = True
                else:
                    words_per_batch[b].append(self.id2word_map.get(tid, "<unk>"))
            if all(done):
                break
        return [" ".join(w) for w in words_per_batch]


class ScaledGRUBaselineModel(nn.Module):
    def __init__(self, world, input_dim):
        super().__init__()
        self.embed = nn.Embedding(world["vocab_size"], D, padding_idx=world["pad"])
        self.encoder = GRUBaselineEncoder(input_dim)
        self.decoder = ScaledGRUBaselineDecoder(self.embed, world["sos"], world["eos"], world["id2word"])

    def forward(self, scene_vec, target_tokens):
        scene_embed = self.encoder(scene_vec)
        return self.decoder(scene_embed, target_tokens)

    @torch.no_grad()
    def describe(self, scene_vec):
        self.eval()
        scene_embed = self.encoder(scene_vec.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(scene_embed)[0]
        self.train()
        return result


# ─────────────────────────────────────────────────────
# 13.4.5 — Dual-scene (paragraph) — Stage5-equivalent at scaled world size
# ─────────────────────────────────────────────────────
def tokenize_paragraph_with_attn_scaled(fact1, fact2, world):
    """Same as tokenize_paragraph_with_attn, using the scaled world's vocab/IDs."""
    obj1_a, rel_a, obj2_a = fact1
    obj1_b, rel_b, obj2_b = fact2
    s1a, c1a, z1a = obj1_a; s2a, c2a, z2a = obj2_a
    s1b, c1b, z1b = obj1_b; s2b, c2b, z2b = obj2_b
    word2id_local = world["word2id"]

    words, attn_targets = [], []

    def add(word, slot=None):
        words.append(word)
        if slot is not None:
            attn_targets.append((len(words) - 1, slot))

    add("the")
    add(z1a, 0); add(c1a, 1); add(s1a, 2)
    add("is")
    for w in rel_a.split():
        add(w, 3)
    add("the")
    add(z2a, 4); add(c2a, 5); add(s2a, 6)
    add(".")

    add("the")
    add(z1b, 7); add(c1b, 8); add(s1b, 9)
    add("is")
    for w in rel_b.split():
        add(w, 10)
    add("the")
    add(z2b, 11); add(c2b, 12); add(s2b, 13)
    add(".")

    tokens = [world["sos"]] + [word2id_local[w] for w in words] + [world["eos"]]
    return tokens, attn_targets


def paragraph_text_scaled(fact1, fact2):
    obj1_a, rel_a, obj2_a = fact1
    obj1_b, rel_b, obj2_b = fact2
    s1a, c1a, z1a = obj1_a; s2a, c2a, z2a = obj2_a
    s1b, c1b, z1b = obj1_b; s2b, c2b, z2b = obj2_b
    return (f"the {z1a} {c1a} {s1a} is {rel_a} the {z2a} {c2a} {s2a} . "
            f"the {z1b} {c1b} {s1b} is {rel_b} the {z2b} {c2b} {s2b} .")


def build_scaled_dataset_dual(fact_pool, world, n_paragraphs):
    """
    A paragraph = two independent facts sampled from the same pool. Using
    normal_test_pairs gives an ordinary paragraph; using ood_test_pairs
    gives a paragraph where both facts involve a rare shape pair — the
    severe-OOD test extended to paragraph level.
    """
    data = []
    for _ in range(n_paragraphs):
        fact1 = random.choice(fact_pool)
        fact2 = random.choice(fact_pool)
        sv1 = encode_scene_scaled(*fact1, world)
        sv2 = encode_scene_scaled(*fact2, world)
        tokens, attn_targets = tokenize_paragraph_with_attn_scaled(fact1, fact2, world)
        text = paragraph_text_scaled(fact1, fact2)
        data.append((sv1, sv2, tokens, attn_targets, text))
    return data


class ScaledStage5Model(nn.Module):
    """Generalized version of Stage5Model — takes world instead of relying on globals."""
    def __init__(self, world):
        super().__init__()
        self.embed = nn.Embedding(world["vocab_size"], D, padding_idx=world["pad"])
        self.slot_encoder = ScaledSlotEncoder(
            world["n_shapes"], world["n_colors"], world["n_sizes"], world["n_relations"]
        )
        self.decoder = ScaledAttentionDecoder(self.embed, world["sos"], world["eos"], world["id2word"])

    def encode_two_facts(self, sv1, sv2):
        memory1, raw1 = self.slot_encoder(sv1, slot_offset=0, return_raw=True)
        memory2, raw2 = self.slot_encoder(sv2, slot_offset=7, return_raw=True)
        memory = torch.cat([memory1, memory2], dim=1)
        return memory, raw1, raw2

    def forward(self, sv1, sv2, target_tokens, return_attn=False):
        memory, raw1, raw2 = self.encode_two_facts(sv1, sv2)
        if return_attn:
            logits, attn = self.decoder(memory, target_tokens, return_attn=True)
            return logits, raw1, raw2, attn
        logits = self.decoder(memory, target_tokens)
        return logits, raw1, raw2

    @torch.no_grad()
    def describe(self, sv1, sv2):
        self.eval()
        memory, _, _ = self.encode_two_facts(sv1.unsqueeze(0).to(DEVICE), sv2.unsqueeze(0).to(DEVICE))
        result = self.decoder.generate(memory, max_len=26)[0]
        self.train()
        return result


class ScaledGRUBaselineModelDual(nn.Module):
    """Generalized version of GRUBaselineModelDual."""
    def __init__(self, world, single_scene_dim):
        super().__init__()
        self.embed = nn.Embedding(world["vocab_size"], D, padding_idx=world["pad"])
        self.encoder = GRUBaselineEncoder(single_scene_dim * 2)
        self.decoder = ScaledGRUBaselineDecoder(self.embed, world["sos"], world["eos"], world["id2word"])

    def forward(self, sv1, sv2, target_tokens):
        combined = torch.cat([sv1, sv2], dim=-1)
        scene_embed = self.encoder(combined)
        return self.decoder(scene_embed, target_tokens)

    @torch.no_grad()
    def describe(self, sv1, sv2):
        self.eval()
        combined = torch.cat([sv1.unsqueeze(0), sv2.unsqueeze(0)], dim=-1).to(DEVICE)
        scene_embed = self.encoder(combined)
        result = self.decoder.generate(scene_embed, max_len=26)[0]
        self.train()
        return result


def _fwd_scaled_full_dual(model, sv1, sv2, tokens):
    logits, raw1, raw2, attn = model(sv1, sv2, tokens, return_attn=True)
    return logits, attn, (raw1, raw2)

def _aux_scaled_full_dual(model, sv1, sv2, raw_pair):
    raw1, raw2 = raw_pair
    return model.slot_encoder.auxiliary_loss(sv1, raw1) + model.slot_encoder.auxiliary_loss(sv2, raw2)

def _fwd_scaled_gru_dual(model, sv1, sv2, tokens):
    logits = model(sv1, sv2, tokens)
    return logits, None, None


def _fwd_scaled_full(model, scenes, tokens):
    logits, raw_slots, attn = model(scenes, tokens, return_attn=True)
    return logits, attn, raw_slots

def _aux_scaled_full(model, scenes, raw_slots):
    return model.slot_encoder.auxiliary_loss(scenes, raw_slots)

def _fwd_scaled_gru(model, scenes, tokens):
    logits = model(scenes, tokens)
    return logits, None, None


# ─────────────────────────────────────────────────────
# 13.5 — Train/eval (same Stage 10 performance optimizations: GPU-resident + vectorized attention loss)
# ─────────────────────────────────────────────────────
def train_scaled_model(model_ctor, train_data, epochs, batch_size, forward_fn,
                        aux_loss_fn=None, attn_sup_weight=0.5, aux_weight=0.15,
                        warmup_epoch=None, pad_id=0):
    if warmup_epoch is None:
        warmup_epoch = max(10, epochs // 6)

    model = model_ctor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    scenes_all = torch.stack([d[0] for d in train_data]).to(DEVICE)
    tokens_all = pad_batch([d[1] for d in train_data]).to(DEVICE)   # PAD=0 always (fixed special-token order)
    max_targets = max((len(d[2]) for d in train_data), default=1)
    pos_idx_all, slot_idx_all = build_attn_supervision_tensors([d[2] for d in train_data], max_targets)
    pos_idx_all = pos_idx_all.to(DEVICE)
    slot_idx_all = slot_idx_all.to(DEVICE)
    N = scenes_all.size(0)
    vocab_size_local = model.embed.num_embeddings

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        warmup_ok = epoch >= warmup_epoch
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            scenes = scenes_all[idx]
            tokens = tokens_all[idx]
            pos_idx = pos_idx_all[idx]
            slot_idx = slot_idx_all[idx]

            logits, attn, raw_slots = forward_fn(model, scenes, tokens)
            target = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size_local), target.reshape(-1), ignore_index=pad_id
            )

            if attn is not None:
                loss = loss + attn_sup_weight * vectorized_attention_loss(attn, pos_idx, slot_idx)

            if raw_slots is not None and aux_loss_fn is not None and warmup_ok:
                aux = aux_loss_fn(model, scenes, raw_slots)
                if aux is not None:
                    loss = loss + aux_weight * aux

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    return model


@torch.no_grad()
def eval_scaled_model(model, data):
    correct = 0
    for scene_vec, tokens, attn_targets, text in data:
        gen = model.describe(scene_vec)
        if gen.strip() == text.strip():
            correct += 1
    return correct / len(data) if data else 0.0


def train_scaled_model_dual(model_ctor, train_data, epochs, batch_size, forward_fn,
                             aux_loss_fn=None, attn_sup_weight=0.5, aux_weight=0.15,
                             warmup_epoch=None, pad_id=0):
    """Same as train_scaled_model — two-scene (Stage5-equivalent) version."""
    if warmup_epoch is None:
        warmup_epoch = max(10, epochs // 6)

    model = model_ctor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    sv1_all = torch.stack([d[0] for d in train_data]).to(DEVICE)
    sv2_all = torch.stack([d[1] for d in train_data]).to(DEVICE)
    tokens_all = pad_batch([d[2] for d in train_data]).to(DEVICE)
    max_targets = max((len(d[3]) for d in train_data), default=1)
    pos_idx_all, slot_idx_all = build_attn_supervision_tensors([d[3] for d in train_data], max_targets)
    pos_idx_all = pos_idx_all.to(DEVICE)
    slot_idx_all = slot_idx_all.to(DEVICE)
    N = sv1_all.size(0)
    vocab_size_local = model.embed.num_embeddings

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        warmup_ok = epoch >= warmup_epoch
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            sv1 = sv1_all[idx]
            sv2 = sv2_all[idx]
            tokens = tokens_all[idx]
            pos_idx = pos_idx_all[idx]
            slot_idx = slot_idx_all[idx]

            logits, attn, raw_slots_pair = forward_fn(model, sv1, sv2, tokens)
            target = tokens[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size_local), target.reshape(-1), ignore_index=pad_id
            )

            if attn is not None:
                loss = loss + attn_sup_weight * vectorized_attention_loss(attn, pos_idx, slot_idx)

            if raw_slots_pair is not None and aux_loss_fn is not None and warmup_ok:
                aux = aux_loss_fn(model, sv1, sv2, raw_slots_pair)
                if aux is not None:
                    loss = loss + aux_weight * aux

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

    return model


@torch.no_grad()
def eval_scaled_model_dual(model, data):
    correct = 0
    for sv1, sv2, tokens, attn_targets, text in data:
        gen = model.describe(sv1, sv2)
        if gen.strip() == text.strip():
            correct += 1
    return correct / len(data) if data else 0.0


# ─────────────────────────────────────────────────────
# 13.6 — Full report: Stage3/Stage5/Stage6-equivalent, accuracy + time + memory
# ─────────────────────────────────────────────────────
def run_world_scaling_experiment(n_shapes=25, n_colors=25, n_sizes=20, n_relations=25,
                                  seed=0, epochs=WORLD_SCALING_EPOCHS, batch_size=WORLD_SCALING_BATCH_SIZE,
                                  n_train_pairs=WORLD_SCALING_N_TRAIN_PAIRS,
                                  n_test_pairs=WORLD_SCALING_N_TEST_PAIRS, n_ood_shapes=None,
                                  n_per_train=2, n_per_test=2,
                                  n_train_paragraphs=N_TRAIN_PARAGRAPHS, n_test_paragraphs=N_HELD_PARAGRAPHS,
                                  models=("Full", "GRU")):
    """
    Full report (accuracy + training time + GPU memory) on a scaled world —
    the same three core tests:
      Stage3-equivalent : plain sentence, ordinary held-out
      Stage5-equivalent : two-sentence paragraph (two independent facts)
      Stage6-equivalent : severe-OOD — shape pairs that never co-occurred at all
    """
    world = build_scaled_world(n_shapes, n_colors, n_sizes, n_relations)
    split = build_scaled_split_with_severe_ood(
        world, seed, n_ood_shapes=n_ood_shapes,
        n_train_pairs=n_train_pairs, n_test_pairs=n_test_pairs,
    )

    # --- Stage3/Stage6-equivalent (single scene) ---
    train_data_local = build_scaled_dataset(split["train_pairs"], world, n_per=n_per_train)
    normal_test_data = build_scaled_dataset(split["normal_test_pairs"], world, n_per=n_per_test)
    ood_test_data = build_scaled_dataset(split["ood_test_pairs"], world, n_per=n_per_test)

    # --- Stage5-equivalent (paragraph, two scenes) ---
    dual_train_data = build_scaled_dataset_dual(split["train_pairs"], world, n_train_paragraphs)
    dual_normal_test_data = build_scaled_dataset_dual(split["normal_test_pairs"], world, n_test_paragraphs)
    dual_ood_test_data = build_scaled_dataset_dual(split["ood_test_pairs"], world, n_test_paragraphs)

    print("=" * 70)
    print("WORLD SCALING EXPERIMENT")
    print("=" * 70)
    print(f"   shapes={n_shapes} colors={n_colors} sizes={n_sizes} relations={n_relations}")
    print(f"   vocab_size={world['vocab_size']} | scene_dim={world['scene_dim']}")
    print(f"   [Stage3-equiv] train={len(train_data_local)} | normal test={len(normal_test_data)} | "
          f"severe-OOD test={len(ood_test_data)}")
    print(f"   [Stage5-equiv] train={len(dual_train_data)} | normal test={len(dual_normal_test_data)} | "
          f"severe-OOD test={len(dual_ood_test_data)}")
    print(f"   OOD shapes (never paired with each other in training): {sorted(split['ood_shapes'])}\n")

    results = {}

    for model_name in models:
        print(f"-- {model_name} --")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = _time.time()

        # --- Stage3/6-equivalent (single scene) ---
        if model_name == "Full":
            model_single = train_scaled_model(
                lambda: ScaledStage3Model(world), train_data_local, epochs, batch_size,
                forward_fn=_fwd_scaled_full, aux_loss_fn=_aux_scaled_full, pad_id=world["pad"],
            )
        elif model_name == "GRU":
            model_single = train_scaled_model(
                lambda: ScaledGRUBaselineModel(world, world["scene_dim"]), train_data_local, epochs, batch_size,
                forward_fn=_fwd_scaled_gru, aux_loss_fn=None, pad_id=world["pad"],
            )
        else:
            raise ValueError(f"unknown model: {model_name}")

        acc_s3 = eval_scaled_model(model_single, normal_test_data)
        acc_s6 = eval_scaled_model(model_single, ood_test_data)

        # --- Stage5-equivalent (paragraph, two scenes) ---
        if model_name == "Full":
            model_dual = train_scaled_model_dual(
                lambda: ScaledStage5Model(world), dual_train_data, epochs, batch_size,
                forward_fn=_fwd_scaled_full_dual, aux_loss_fn=_aux_scaled_full_dual, pad_id=world["pad"],
            )
        else:
            model_dual = train_scaled_model_dual(
                lambda: ScaledGRUBaselineModelDual(world, world["scene_dim"]), dual_train_data, epochs, batch_size,
                forward_fn=_fwd_scaled_gru_dual, aux_loss_fn=None, pad_id=world["pad"],
            )

        acc_s5 = eval_scaled_model_dual(model_dual, dual_normal_test_data)
        acc_s5_ood = eval_scaled_model_dual(model_dual, dual_ood_test_data)

        train_time = _time.time() - t0
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None

        results[model_name] = {
            "stage3_acc": acc_s3, "stage5_acc": acc_s5, "stage6_acc": acc_s6,
            "stage5_ood_acc": acc_s5_ood,
            "train_time_sec": train_time, "peak_mem_mb": peak_mem_mb,
        }
        mem_str = f"{peak_mem_mb:.1f} MB" if peak_mem_mb is not None else "N/A (CPU)"
        print(f"   stage3_acc={acc_s3:.3f} | stage5_acc={acc_s5:.3f} | stage6_acc(severe-OOD)={acc_s6:.3f} | "
              f"stage5_severe_ood_acc={acc_s5_ood:.3f}")
        print(f"   train_time={train_time:.1f}s (Stage3/6 + Stage5 combined) | peak_gpu_mem={mem_str}\n")

    print("=" * 70)
    print("WORLD SCALING — FINAL REPORT")
    print("=" * 70)
    print(f"  {'Model':8s} | {'Stage3':7s} | {'Stage5':7s} | {'Stage6(OOD)':11s} | "
          f"{'Stage5-OOD':10s} | {'Time(s)':8s} | {'Mem(MB)':8s}")
    print("  " + "-" * 78)
    for name, r in results.items():
        mem_str = f"{r['peak_mem_mb']:.1f}" if r["peak_mem_mb"] is not None else "N/A"
        print(f"  {name:8s} | {r['stage3_acc']:.3f}   | {r['stage5_acc']:.3f}   | {r['stage6_acc']:.3f}       "
              f"| {r['stage5_ood_acc']:.3f}      | {r['train_time_sec']:.1f}    | {mem_str}")
    print()

    return world, split, results


def run_world_scaling_curve(configs=None, seed=0, epochs=WORLD_SCALING_EPOCHS,
                             batch_size=WORLD_SCALING_BATCH_SIZE, **kwargs):
    """
    Runs run_world_scaling_experiment across more than one world size — a
    performance/time/memory curve rather than a single data point. Starts
    from the original size (4x4x2x4) as a reference and scales up to the
    requested size (25x25x20x25). Any extra **kwargs (e.g.
    n_train_paragraphs) pass straight through to every experiment — the
    same setting applies to every world size equally.
    """
    if configs is None:
        configs = WORLD_SCALING_CONFIGS

    all_results = {}
    for cfg_dict in configs:
        label = f"{cfg_dict['n_shapes']}x{cfg_dict['n_colors']}x{cfg_dict['n_sizes']}x{cfg_dict['n_relations']}"
        print(f"\n\n### WORLD SIZE: {label} ###")
        world, split, results = run_world_scaling_experiment(
            **cfg_dict, seed=seed, epochs=epochs, batch_size=batch_size, **kwargs
        )
        all_results[label] = results

    print("=" * 70)
    print("WORLD SCALING CURVE — SUMMARY ACROSS SIZES")
    print("=" * 70)
    for label, res in all_results.items():
        for model_name, r in res.items():
            mem_str = f"{r['peak_mem_mb']:.1f}MB" if r["peak_mem_mb"] is not None else "N/A"
            print(f"  size={label:18s} model={model_name:6s} stage3={r['stage3_acc']:.3f} "
                  f"stage5={r['stage5_acc']:.3f} stage6(OOD)={r['stage6_acc']:.3f} "
                  f"stage5-OOD={r['stage5_ood_acc']:.3f} time={r['train_time_sec']:.1f}s mem={mem_str}")
    print()
    return all_results


# ═══════════════════════════════════════════════════════════════
# MAIN — Stage 13 only (World Scaling)
# ═══════════════════════════════════════════════════════════════
"""
Stages 1-12 (direct classification, learned embeddings, seq2seq+attention,
paraphrase understanding, paragraphs, compositionality, GRU baseline,
multi-split benchmark, ablation study, multi-seed evaluation, Transformer
baselines) are already validated from prior runs. Re-running them with the
same settings would not add new information, so they are not invoked here.

To re-run any of them, every function is still defined and directly
callable:
  train_stage1() ... train_stage5(), train_compositionality_test(),
  run_all_baselines(), run_multi_split_benchmark(), run_ablation_study(),
  run_multi_seed_benchmark(...), run_transformer_baselines(...),
  run_stage8_comparison(...)

The focus here is the open question: do these results hold as the world
scales up? We run the same three core tests (Stage3/Stage5/Stage6-
equivalent) on a small world (4x4x2x4 — matching the Stage 1-12 reference
size) and a large world (25x25x20x25), recording accuracy, training time,
and GPU memory for each, for a direct comparison.
"""
if __name__ == "__main__":
    import traceback

    print("\n" + "=" * 70)
    print("  WORLD SCALING — GROUNDED FEW-SHOT ENGLISH LEARNER")
    print("=" * 70 + "\n")
    print("Stages 1-12 are already validated from prior runs and are not")
    print("repeated here. Focus: accuracy + time + memory, small world vs. large world.\n")

    run_results = {}

    try:
        curve_results = run_world_scaling_curve(
            configs=WORLD_SCALING_CONFIGS,
            epochs=WORLD_SCALING_EPOCHS, batch_size=WORLD_SCALING_BATCH_SIZE,
            n_train_paragraphs=WORLD_SCALING_N_TRAIN_PARAGRAPHS,
        )
        run_results["stage13"] = f"done: {curve_results}"
    except Exception:
        print("\nSTAGE 13 FAILED:")
        traceback.print_exc()
        run_results["stage13"] = "failed"

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for stage, status in run_results.items():
        print(f"  {stage}: {status}")
    print()
