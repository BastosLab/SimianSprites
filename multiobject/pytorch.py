import cv2 as cv
import functools
import math
import numpy as np
import os
import torch
from torch.distributions.uniform import Uniform
from torch.nn.functional import affine_grid, grid_sample, normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

class SpritesVideo(torch.nn.Module):
    PUNCH_OUT_COLOR = 1

    def __init__(self, frame_size, sprites, vs, xs, rf=None):
        super().__init__()

        assert xs.shape[1:] == vs.shape[1:]
        self.register_buffer('xs', xs)
        self.register_buffer('vs', vs)

        if rf is None:
            rf = torch.tensor((torch.nan, torch.nan, torch.nan, torch.nan,
                                torch.nan))
        self.register_buffer('rf', rf.to(dtype=torch.float32))

        assert sprites.shape[0] == self.num_sprites
        self.frame_size = frame_size
        self.register_buffer('sprites',
                             torch.from_numpy(sprites.astype('float32')))

        self.register_buffer('occlusions', torch.zeros(self.timesteps, 3))

    @staticmethod
    def degrees_to_coords(x, y):
        return (x / SimSpritesVideo.SCREEN_HALFWIDTH_DEGREES,
                y / SimSpritesVideo.SCREEN_HALFHEIGHT_DEGREES)

    @property
    def egocentric(self):
        dims = torch.tensor([SimSpritesVideo.SCREEN_HALFWIDTH_DEGREES,
                             SimSpritesVideo.SCREEN_HALFHEIGHT_DEGREES])
        dims = dims.expand(1, 1, 2)
        return (self.xs * 2 * dims, self.vs * 2 * dims)

    @staticmethod
    def coords_to_pixels(x, y):
        return (x * (SimSpritesVideo.SCREEN_RES[0] / 2),
                y * (SimSpritesVideo.SCREEN_RES[1] / 2))

    @property
    def num_sprites(self):
        return self.xs.shape[0]

    def punch_frame(self, frame, t, punchout=True):
        frame = frame.numpy().transpose(1, 0, 2)

        mask = np.zeros(frame.shape, np.uint8)
        (x, y, rx, ry, theta) = self.rf
        x, y = SpritesVideo.coords_to_pixels(x, y)
        x = int(torch.round(SimSpritesVideo.SCREEN_RES[0] / 2 + x))
        y = int(torch.round(SimSpritesVideo.SCREEN_RES[1] / 2 - y))
        rx, ry = SpritesVideo.coords_to_pixels(rx, ry)
        rx, ry = int(torch.round(rx)), int(torch.round(ry))
        c = SpritesVideo.PUNCH_OUT_COLOR

        mask = cv.ellipse(mask, (x, y), (rx, ry), torch.rad2deg(theta).item(),
                          0, 360, (c, c, c), -1)

        if punchout:
            frame = np.where(mask > 0, mask, frame)
        else:
            frame = np.where(mask > 0, frame, mask + c)
        unoccluded = (frame > 1)[:, :, 0].sum() / math.prod(self.sprite_shape)
        self.occlusions[t, 1 + int(punchout)] = 1. - unoccluded
        return frame.transpose(1, 0, 2)

    def render(self, punchout=None):
        video = []
        for t in range(self.timesteps):
            frame = self.render_frame(t)
            if punchout is not None:
                frame, _ = self.punch_frame(frame, t, punchout)
            video.append(frame)
        return torch.stack(video, dim=0).clamp(min=0, max=255).to(torch.uint8)

    def render_frame(self, t):
        scaling = self.scaling().unsqueeze(dim=0)
        translation = self.translation()

        translation = (self.xs[:, t] * translation).unsqueeze(dim=-1)
        transforms = torch.cat((scaling, translation), dim=-1)

        grids = affine_grid(transforms, torch.Size((len(self.sprites),
                                                    self.sprites.shape[-1],
                                                    self.frame_size[1],
                                                    self.frame_size[0])),
                            align_corners=False)
        src = self.sprites.movedim(-1, 1)
        dest = grid_sample(src, grids, mode='nearest', align_corners=False)
        frame = dest.transpose(-1, -2).movedim(1, -1)
        return frame.sum(dim=0)

    @functools.cache
    def scaling(self):
        return torch.eye(2) * torch.tensor(self.frame_size) /\
               torch.tensor(self.sprite_shape)

    @property
    def sprite_shape(self):
        if not torch.isnan(self.rf).any():
            rx, ry = self.rf[2:4]
            rx, ry = SpritesVideo.coords_to_pixels(rx, ry)
            sprite_side = int((min(rx, ry) / math.sqrt(2)).round())
            return torch.Size([sprite_side, sprite_side])
        return self.sprites.shape[1:3]

    @property
    def timesteps(self):
        return self.xs.shape[1]

    @functools.cache
    def translation(self):
        frame_size = torch.tensor(self.frame_size)
        sprite_shape = torch.tensor(self.sprite_shape)
        translation = -(frame_size - sprite_shape) / sprite_shape
        translation[1] *= -1
        return translation

    def write(self, path, punches=True):
        frames = self.render()

        writer = cv.VideoWriter(path + '_full.mp4',
                                cv.VideoWriter_fourcc(*"mp4v"),
                                SimSpritesVideo.FPS, tuple(self.frame_size),
                                True)
        for t in range(frames.shape[0]):
            frame = cv.cvtColor(frames[t].transpose(0, 1).numpy(),
                                cv.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        if punches:
            fname = os.path.basename(path)
            inpath = os.path.dirname(os.path.dirname(path)) + '/punch_in/'
            if not os.path.exists(inpath):
                os.makedirs(inpath)
            inpath = inpath + fname + "_in.mp4"
            writer = cv.VideoWriter(inpath, cv.VideoWriter_fourcc(*"mp4v"),
                                    SimSpritesVideo.FPS, tuple(self.frame_size),
                                    True)
            for t in range(frames.shape[0]):
                frame = self.punch_frame(frames[t], t, False).transpose(1, 0, 2)
                writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            writer.release()

            outpath = os.path.dirname(os.path.dirname(path)) + '/punch_out/'
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            outpath = outpath + fname + "_out.mp4"
            writer = cv.VideoWriter(outpath, cv.VideoWriter_fourcc(*"mp4v"),
                                    SimSpritesVideo.FPS, tuple(self.frame_size),
                                    True)
            for t in range(frames.shape[0]):
                frame = self.punch_frame(frames[t], t, True).transpose(1, 0, 2)
                writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            writer.release()

        torch.save(self, path + '.pt')

class SimSpritesVideo:
    DISTANCE_TO_SCREEN = 106
    FPS = 60
    SCREEN_DIMS = (100, 62)
    SCREEN_HALFWIDTH_DEGREES = np.degrees(np.arctan(
        SCREEN_DIMS[0] / 2 / DISTANCE_TO_SCREEN
    )).astype("float32")
    SCREEN_HALFHEIGHT_DEGREES = np.degrees(np.arctan(
        SCREEN_DIMS[1] / 2 / DISTANCE_TO_SCREEN
    )).astype("float32")
    SCREEN_RES = (1920, 1080)

    def __init__(self, timesteps, frame_sizes, rf=None):
        self.rf = torch.tensor(rf).to(torch.float32) if rf is not None else None
        assert self.rf is None or self.rf.shape == (5,)
        self.timesteps = timesteps
        self.frame_sizes = torch.tensor(frame_sizes)

    @torch.no_grad()
    def sim_video(self, sprites, x0, v0):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        x0 = torch.from_numpy(x0.astype('float32'))
        v0 = torch.from_numpy(v0.astype('float32'))
        xs, vs = self.sim_trajectories(len(sprites), x0, v0)
        return SpritesVideo(torch.Size(self.frame_sizes), sprites, vs, xs,
                            rf=self.rf)

    def sim_trajectories(self, num_tjs, x0, v0):
        xs = []
        vs = []
        for i in range(num_tjs):
            x, v = self.sim_trajectory(x0[i], v0[i])
            xs.append(x)
            vs.append(v)
        return torch.stack(xs, dim=0), torch.stack(vs, dim=0)

    def sim_trajectory(self, init_xs, init_vs):
        ''' Generate a random sequence of a sprite '''
        X = torch.zeros((self.timesteps, 2))
        V = torch.zeros((self.timesteps, 2))
        X[0] = init_xs
        V[0] = init_vs
        for t in range(0, self.timesteps -1):
            X_new = X[t] + V[t] / SimSpritesVideo.FPS
            V_new = V[t]

            if X_new[0] < -1.0:
                X_new[0] = -1.0 + torch.abs(-1.0 - X_new[0])
                V_new[0] = - V_new[0]
            if X_new[0] > 1.0:
                X_new[0] = 1.0 - torch.abs(X_new[0] - 1.0)
                V_new[0] = - V_new[0]
            if X_new[1] < -1.0:
                X_new[1] = -1.0 + torch.abs(-1.0 - X_new[1])
                V_new[1] = - V_new[1]
            if X_new[1] > 1.0:
                X_new[1] = 1.0 - torch.abs(X_new[1] - 1.0)
                V_new[1] = - V_new[1]
            V[t+1] = V_new
            X[t+1] = X_new
        return X, V

class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        # Max label length in this batch
        # max_len[k] is the maximum length (in batch) of the label with name k
        # If at the end max_len[k] is -1, labels k are (probably all) scalars
        max_len = {k: -1 for k in keys}

        # If a label has more than 1 dimension, the padded tensor cannot simply
        # have size (batch, max_len). Whenever the length is >0 (i.e. the sequence
        # is not empty, store trailing dimensions. At the end if 1) all sequences
        # (in the batch, and for this label) are empty, or 2) this label is not
        # a sequence (scalar), then the trailing dims are None.
        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        # For each item in the batch, take each key and pad the corresponding
        # value (label) so we can call the default collate function
        pad = MultiObjectDataLoader._pad_tensor
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue
                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = pad(batch[i][1][k], size)

        return default_collate(batch)

    @staticmethod
    def _pad_tensor(x, size, value=None):
        assert isinstance(x, torch.Tensor)
        input_size = len(x)
        if value is None:
            value = float('nan')

        # Copy input tensor into a tensor filled with specified value
        # Convert everything to float, not ideal but it's robust
        out = torch.zeros(*size, dtype=torch.float)
        out.fill_(value)
        if input_size > 0:  # only if at least one element in the sequence
            out[:input_size] = x.float()
        return out


class MultiObjectDataset(Dataset):

    def __init__(self, data_path, train, split=0.9):
        super().__init__()

        # Load data
        data = np.load(data_path, allow_pickle=True)

        # Rescale images and permute dimensions
        x = np.asarray(data['x'], dtype=np.float32) / 255
        x = np.transpose(x, [0, 3, 1, 2])  # batch, channels, h, w

        # Get labels
        labels = data['labels'].item()

        # Split train and test
        split = int(split * len(x))
        if train:
            indices = range(split)
        else:
            indices = range(split, len(x))

        # From numpy/ndarray to torch tensors (labels are lists of tensors as
        # they might have different sizes)
        self.x = torch.from_numpy(x[indices])
        self.labels = self._labels_to_tensorlist(labels, indices)

    @staticmethod
    def _labels_to_tensorlist(labels, indices):
        out = {k: [] for k in labels.keys()}
        for i in indices:
            for k in labels.keys():
                t = labels[k][i]
                t = torch.as_tensor(t)
                out[k].append(t)
        return out

    def __getitem__(self, index):
        x = self.x[index]
        labels = {k: self.labels[k][index] for k in self.labels.keys()}
        return x, labels

    def __len__(self):
        return self.x.size(0)
