from cvauth.arcface.utils import draw_box_name


def visualize_face(vis, bbox, score, name, args_score):
    if args_score:
        vis = draw_box_name(bbox, name + '_{:.2f}'.format(score), vis)
    else:
        vis = draw_box_name(bbox, name, vis)

    return vis
