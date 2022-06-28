import torch


class IoU:
    def __init__(self):
        pass

    def __call__(self, bboxes1, bboxes2, eps=1e-6):
        if bboxes1.size()[1] == 5:
            bboxes1 = bboxes1[:, :4]
        if bboxes2.size()[1] == 5:
            bboxes2 = bboxes2[:, :4]

        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)

        if rows * cols == 0:
            return bboxes1.new(batch_shape + (rows, cols))

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = rb - lt
        wh = wh.clamp(0, None)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union

        return ious
