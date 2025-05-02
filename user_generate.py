import argparse
import json
import os
import subprocess
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import UserDataset
import pickle
from models.model import SDT_Generator
import tqdm
from utils.util import writeCache, dxdynp_to_list, coords_render
import lmdb


def main(opt):
    """load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """setup data_loader instances"""
    test_dataset = UserDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, opt.style_path
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.IMS_PER_BATCH,
        shuffle=True,
        sampler=None,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
    )
    # 保证了输出目录存在
    os.makedirs(os.path.join(opt.save_dir), exist_ok=True)

    """build model architecture"""
    model = SDT_Generator(
        num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
        num_head_layers=cfg.MODEL.NUM_HEAD_LAYERS,
        wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
        gly_dec_layers=cfg.MODEL.GLY_DEC_LAYERS,
    ).to("cuda")
    if len(opt.pretrained_model) > 0:
        model_weight = torch.load(opt.pretrained_model)
        model.load_state_dict(model_weight)
        print("load pretrained model from {}".format(opt.pretrained_model))
    else:
        raise IOError("input the correct checkpoint path")
    model.eval()

    """setup the dataloader"""
    batch_samples = len(test_loader)
    data_iter = iter(test_loader)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(batch_samples)):

            data = next(data_iter)
            # prepare input
            img_list, char_img, char = (
                data["img_list"].cuda(),
                data["char_img"].cuda(),
                data["char"],
            )
            preds = model.inference(img_list, char_img, 120)
            bs = char_img.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
            preds = preds.detach().cpu().numpy()

            # 计算平均大小
            total_size = 0
            total_count = len(preds)
            for i, _ in enumerate(preds):
                # 平移坐标。如果在这里做坐标平移的话会导致绘图错误，可能是coords_render函数中对坐标进行了修改
                """Render the character images by connecting the coordinates"""
                save_path = os.path.join(opt.save_dir, char[i] + ".txt")
                # 这里面会对坐标修改以得到绘图的坐标点，所以我不能使用这个函数
                # 还必须有这个函数，不然得到的坐标点全是乱的。。。
                # 一定要有这个函数是因为这里面将相对坐标转换为了绝对坐标
                # 这里传入preds[i].copy()以免影响preds[i]的值
                # 这里面之后计算绝对坐标会影响preds，所以还是可以直接传入preds[i]的
                sk_pil, coords, average_size = coords_render(
                    preds[i],
                    split=True,
                    width=256,
                    height=256,
                    thickness=8,
                    board=1,
                )
                # print("the coordinates are:", coords)
                # 同时保存图片
                sk_pil.save(save_path.replace(".txt", ".png"))
                # 计算绝对坐标
                ################################################
                total_size += average_size
                ################################################
                try:
                    # 将所有的坐标都保存到output中
                    with open(save_path, "w") as f:
                        f.write(str(coords))
                except:
                    print("error. %s, %s" % (save_path, char[i]))
                # 仅保存十张图片用于调试
                if i == 10:
                    break
            # 这里也设置为10，因为只生成了十个汉字
            overall_average_size = total_size / 10
            break
    # 将平均大小返回，在makefile中接收或者直接在这里保存到json中也行
    print("Average size of generated characters: ", overall_average_size)
    return overall_average_size


if __name__ == "__main__":
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        default="configs/CHINESE_USER.yml",
        help="Config file for training (and optionally testing)",
    )
    # 选择将所有的坐标信息都保存到output中，原来的值为Generated/Chinese_User
    parser.add_argument(
        "--dir",
        dest="save_dir",
        default="../output/SDT",
        help="target dir for storing the generated characters",
    )
    parser.add_argument(
        "--pretrained_model",
        dest="pretrained_model",
        default="",
        required=True,
        help="continue train model",
    )
    parser.add_argument(
        "--style_path",
        dest="style_path",
        default="style_samples",
        help="dir of style samples",
    )
    opt = parser.parse_args()
    main(opt)
