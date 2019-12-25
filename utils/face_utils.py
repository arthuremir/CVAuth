from cvauth.arcface.utils import draw_box_name


def visualize_face(vis, faces, names, args_score):
    bboxes, results, score = faces[0]

    for idx, bbox in enumerate(bboxes):
        if args_score:
            vis = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), vis)
        else:
            vis = draw_box_name(bbox, names[results[idx] + 1], vis)

    return vis
