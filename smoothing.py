import glob
import json
import math
import os
import pickle
import shutil
import time
from pprint import pprint

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ### GLOBAL PARAMETERS

# These need not be changed. FPS can be changed to slow down or speed up visualization for further inspection.
FPS_INPUT = 29.97
FPS = FPS_INPUT  # FPS of output video, in principle should correspond to FPS of input video
# Can be modified during experimentation
VIDEO_SHAPE = (2160, 3840)  # original video shape in format (H, W)
IM_SRC = os.path.join(os.path.dirname(__file__), "smoothing",
                      "pitch_coords_ek.png")  # pitch template to map onto

# color palette for visualized objects
TEAM_PALETTE_DICT = {
    "player": "#984ea3",
    "referee": "#57d3db",
    "other": "#4daf4a",
    "goalkeeper": "#ffd92f",
    "ball": "#66c2a5",
    "person": "#984ea3",
    "shirt_number": "#984ea3",
}
TEAM_COLOR_DICT = {0: (255, 240, 248), 1: (163, 47, 67)}


def hex_to_rgb(hex_tuple):
    """
    Transform hex-encoded color to RGB
    """
    h = hex_tuple.lstrip("#")
    rgb = tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))
    return rgb


def run_tactical_module(df_tactical, root, video_shape, fps):
    """
    Run tactical visualization based on processed input
    """
    parameters = {
        "SIZE_PLAYER": 8,
        "SIZE_BALL": 4,
        "sequence": "tactical",
        "fps": fps,
        "video_format": ".mp4",
    }
    colors = {
        "goalkeeper": hex_to_rgb(TEAM_PALETTE_DICT.get("goalkeeper"))[::-1],
        "referee": hex_to_rgb(TEAM_PALETTE_DICT.get("referee"))[::-1],
        "ball": (0, 0, 0),
        "white": (255, 255, 255),
    }
    # define module
    module_tactics = ModuleTacticalBase(
        root=root, shape=video_shape, team_color_dict=TEAM_COLOR_DICT, fps=fps,
    )
    # visualize tactical view
    module_tactics.make_visualization(
        df_tactical, parameters, cv2.imread(IM_SRC), colors
    )
    return


class ModuleTacticalBase(object):
    def __init__(
        self, root, shape, team_color_dict, fps=30.0,
    ):
        self.root = root
        self.shape = shape
        self.team_color_dict = team_color_dict
        self.fps = fps

    def make_visualization(
        self, df, parameters, im_src, colors, save_video=True,
    ):
        # preparing df
        print("Preparing df for visualization")
        df_players = df.loc[df["category"] == "player"]
        df_goalkeeper = df.loc[df["category"] == "goalkeeper"]
        df_referee = df.loc[df["category"] == "referee"]
        df_ball = df.loc[df["category"] == "ball"]
        # gather team color dicts
        print("Done")
        print("Start visualization for " + parameters.get("sequence"))

        height, width, _ = im_src.shape
        size = (width, height)
        if save_video:
            out = cv2.VideoWriter(
                os.path.join(self.root, "video_tactical.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                parameters.get("fps"),
                size,
            )

        for i in tqdm(df.frame_num.unique()):
            image = im_src.copy()

            df_tmp_players = df_players.loc[(df_players["frame_num"] == i)][
                ["xh", "yh", "team", "track_id", "speed"]
            ].values.reshape(-1, 1, 5)

            df_tmp_goalkeeper = df_goalkeeper.loc[df_goalkeeper["frame_num"] == i][
                ["xh", "yh"]
            ].values.reshape(-1, 1, 2)
            df_tmp_referee = df_referee.loc[df_referee["frame_num"] == i][
                ["xh", "yh"]
            ].values.reshape(-1, 1, 2)
            df_tmp_ball = df_ball.loc[df_ball["frame_num"] == i][
                ["xh", "yh"]
            ].values.reshape(-1, 1, 2)

            for player in range(len(df_tmp_players)):
                if df_tmp_players[player, 0, 2]:
                    team_color = "red"
                else:
                    team_color = "blue"

                player_loc = (
                    df_tmp_players[player, 0, 0].astype(int),
                    df_tmp_players[player, 0, 1].astype(int),
                )
                cv2.circle(
                    image,
                    player_loc,
                    parameters.get("SIZE_PLAYER"),
                    self.team_color_dict.get(df_tmp_players[player, 0, 2]),
                    -1,
                )
                font = cv2.FONT_HERSHEY_SIMPLEX

                fontScale = 0.5
                blue_color = (255, 0, 0)
                thickness = 1
                text = f"%d, speed: %s" % (df_tmp_players[player, 0, 3].astype(
                    int), "~"*(df_tmp_players[player, 0, 4].astype(int)//10))
                cv2.putText(image, text, player_loc, font,
                            fontScale, blue_color, thickness, cv2.LINE_AA)
            for ball in range(len(df_tmp_ball)):
                cv2.circle(
                    image,
                    (
                        df_tmp_ball[ball, 0, 0].astype(int),
                        df_tmp_ball[ball, 0, 1].astype(int),
                    ),
                    parameters.get("SIZE_BALL"),
                    colors.get("ball"),
                    -1,
                )
            for goalkeeper in range(len(df_tmp_goalkeeper)):
                cv2.circle(
                    image,
                    (
                        df_tmp_goalkeeper[goalkeeper, 0, 0].astype(int),
                        df_tmp_goalkeeper[goalkeeper, 0, 1].astype(int),
                    ),
                    parameters.get("SIZE_PLAYER"),
                    colors.get("goalkeeper"),
                    -1,
                )
            for referee in range(len(df_tmp_referee)):
                cv2.circle(
                    image,
                    (
                        df_tmp_referee[referee, 0, 0].astype(int),
                        df_tmp_referee[referee, 0, 1].astype(int),
                    ),
                    parameters.get("SIZE_PLAYER"),
                    colors.get("referee"),
                    -1,
                )

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            blue_color = (255, 0, 0)
            thickness = 2
            cv2.putText(image, f'Frame %d' % i, org, font,
                        fontScale, blue_color, thickness, cv2.LINE_AA)

            if save_video:
                out.write(image)
        out.release()
        print("Done")
        return


# DATA PREPARATOR class
# This one should be modified to improve tactical mapping quality. Smoothing logic can be applied in the apply_homography method.
# This is the part, where image coordinates are mapped onto the pitch plane.

# Currently, the mapping is done in the most basic manner.

# Mapping logic:
# 1. xc_ corresponds to the center of gravity for each person in the X-axis, here it is the middle coordinate
# 2. yc_ corresponds to the maximum coordinate on the Y-axis in the image space, which corresponds to the feet of the player

# Smoothing approaches
# Two ways of smoothing can be used:
# 1. apply smoothing on the raw image coordinates: (x0, y0, x1, y1)
# 2. apply smoothing on the coordinates after mapping onto the pitch plane: (xh, yh)


class DataPreparator(object):
    def __init__(self):
        pass

    def initial_prep(self, df_detection, df_homography):
        """
        Perform initial preparation for homography mapping
        """
        df_detection = df_detection.copy()
        df_homography = df_homography.copy()
        df_homography.rename(
            columns={"frame_index": "file_name"}, inplace=True)
        df_homography["h8"] = df_homography["h8"].fillna(1)
        df_homography = df_homography.dropna().reset_index(drop=True)
        df_detection = df_detection.sort_values("score", ascending=False)
        df_detection = (
            df_detection.groupby(
                ["file_name", "detection_id"]).first().reset_index()
        )
        df_merged = df_detection.merge(
            df_homography, on=["file_name"], how="inner")
        df_merged["frame_num"] = pd.factorize(df_merged["file_name"])[0]
        df_merged["track_id"] = df_merged["track_id"].dropna().astype(
            "int", errors="ignore")

        df_merged["xc_"] = (df_merged["x0"] + df_merged["x1"]) / 2
        df_merged["yc_"] = df_merged["y1"]
        df_merged["bb_height"] = df_merged["y1"] - df_merged["y0"]

        return self.subset_input(df_merged)

    def interpolate_missing_frames(self, df_track):

        category = df_track.iloc[0]["category"]
        track_id = df_track.iloc[0]["track_id"]

        min_frame = df_track["frame_num"].min()
        max_frame = df_track["frame_num"].max()

        for frame in range(min_frame, max_frame):
            if df_track.loc[df_track["frame_num"] == frame, :].empty:
                df_track = df_track.append(
                    {"frame_num": frame, "track_id": track_id, "category": category}, ignore_index=True)

        df_track = df_track.sort_values(
            "frame_num", ignore_index=True)
        df_track = df_track.interpolate()

        return df_track

    def correct_feet_occlusion(self, df_track, window_size: int = int(FPS_INPUT)):

        df_track["median_bb_height"] = df_track["bb_height"].rolling(
            window_size, min_periods=1).median()

        df_track["median_bb_height"].fillna(
            df_track["bb_height"], inplace=True)
        df_track["yc_corrected"] = df_track["y0"] + \
            df_track["median_bb_height"]
        return df_track

    def ewm(self, df_track, alpha=0.4, adjust=False):
        df_track["xc_smoothed"] = df_track["xc_"].ewm(alpha=alpha).mean()
        df_track["yc_smoothed"] = df_track["yc_corrected"].ewm(
            alpha=alpha).mean()
        return df_track

    def add_speed(self, df_track):
        df_track["speed"] = FPS_INPUT*((df_track["xc_smoothed"].diff() * 640 / VIDEO_SHAPE[1]).pow(
            2) + (df_track["xc_smoothed"].diff() * 320 / VIDEO_SHAPE[0]).pow(2)).pow(.5).fillna(0)
        return df_track

    def smooth_tracks(self, df_merged):
        tracks_dfs = [df_merged.loc[df_merged["track_id"]
                                    == np.NaN, :]]
        for track_id in df_merged["track_id"].dropna().unique():
            df_track = df_merged.loc[df_merged["track_id"]
                                     == track_id, :].copy()

            df_track = self.correct_feet_occlusion(df_track)
            df_track = self.interpolate_missing_frames(df_track)
            df_track = self.ewm(df_track)

            df_track = self.add_speed(df_track)
            tracks_dfs.append(df_track)

        return pd.concat(tracks_dfs, ignore_index=True, sort=False)

    def prepare_points_for_homography(self, df_merged, shape):
        # scaling the coordinates to 640x320 (this is due to the model output resolutions)
        df_merged["xc"] = df_merged["xc_smoothed"] * 640 / shape[1]
        df_merged["yc"] = df_merged["yc_smoothed"] * 320 / shape[0]
        assert (np.sum(pd.isnull(df_merged["yc"]))) == 0
        return df_merged

    def apply_homography(self, df_merged):

        df_merged["xh"] = (
            df_merged["h0"] * df_merged["xc"]
            + df_merged["h1"] * df_merged["yc"]
            + df_merged["h2"]
        ) / (
            df_merged["h6"] * df_merged["xc"]
            + df_merged["h7"] * df_merged["yc"]
            + df_merged["h8"]
        )
        df_merged["yh"] = (
            df_merged["h3"] * df_merged["xc"]
            + df_merged["h4"] * df_merged["yc"]
            + df_merged["h5"]
        ) / (
            df_merged["h6"] * df_merged["xc"]
            + df_merged["h7"] * df_merged["yc"]
            + df_merged["h8"]
        )
        # second way of smoothing can be applied here
        # try applying smoothing methods to (xh, yh)
        # df_smoothed = ...
        assert (np.sum(pd.isnull(df_merged["xh"]))) == 0
        assert (np.sum(pd.isnull(df_merged["yh"]))) == 0
        return df_merged
        # return smoothed DF
        # return df_smoothed

    def subset_input(self, df):
        """
        Subset columns for visualization preparation
        """
        df = df.loc[
            :,
            [
                "frame_num",
                "xc_",
                "y0",
                "y1",
                "yc_",
                "bb_height",
                "category",
                "team",
                "number",
                "track_id",
                "h0",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "h7",
                "h8",
            ],
        ]
        return df

    def subset_output(self, df_merged):
        """
        Subset columns for visualization preparation
        """
        df = df_merged.loc[
            :,
            [
                "frame_num",
                "xh",
                "yh",
                "category",
                "team",
                "number",
                "track_id",
                "speed",
            ],
        ]
        return df


ROOT_SRC = "./smoothing/"
ROOT_DST = "./"

df_det = pd.read_csv(os.path.join(ROOT_SRC, "merged_df_agg.csv"))
df_hom = pd.read_csv(os.path.join(ROOT_SRC, "hom_smooth.csv"))

preparator = DataPreparator()
df_merged = preparator.initial_prep(df_det, df_hom)
df_merged.to_csv("df_merged.csv")

df_merged = preparator.smooth_tracks(df_merged)


df_merged = preparator.prepare_points_for_homography(df_merged, VIDEO_SHAPE)
df_merged_hom = preparator.apply_homography(df_merged)
df_merged_hom.to_csv("df_merged_hom.csv")
df_tactical = preparator.subset_output(df_merged_hom)

run_tactical_module(df_tactical, ROOT_DST, VIDEO_SHAPE, FPS)
