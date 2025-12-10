import sys
import os
import random
import json
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from typing import Dict

# 顔認識モデルの初期化
mp_face_mesh = mp.solutions.face_mesh

# 画像フォルダのパス
PHOTO_DIR = "./photo"  # Docker内の構成に合わせて "/app/photo" でもOK

# ---------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------
def _calc_dist(p1, p2, w, h):
    """2点間のユークリッド距離を計算"""
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _normalize(value, min_v, max_v):
    """値を0.0〜1.0の範囲に正規化してクリップする"""
    return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.0))

def _dist_point_to_line(p_pt, p_start, p_end, w, h):
    """点から直線への距離を計算（眉のカーブ判定用）"""
    px, py = p_pt.x * w, p_pt.y * h
    sx, sy = p_start.x * w, p_start.y * h
    ex, ey = p_end.x * w, p_end.y * h
    
    line_len = np.sqrt((ex - sx)**2 + (ey - sy)**2)
    if line_len == 0: return 0
    cross_prod = abs((ex - sx)*(sy - py) - (sx - px)*(ey - sy))
    return cross_prod / line_len

# ---------------------------------------------------------
# 解析ロジック本体
# ---------------------------------------------------------
def analyze_image(image_path: str) -> Dict[str, float]:
    """画像パスを受け取り、解析結果(0.0-1.0)を返す"""
    
    # 画像読み込み
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        # 画像が開けない場合はエラーを出力して終了せず、Noneを返す等のハンドリングも可
        # ここではシンプルにエラー出力してデフォルト値を返す動きにします
        print(f"Error: 画像を開けませんでした ({image_path}) - {e}", file=sys.stderr)
        sys.exit(1)

    # OpenCV形式への変換
    img_np = np.array(pil_image)
    if img_np.shape[-1] == 4: # RGBA -> RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    img_rgb = img_np
    if len(img_np.shape) == 2: # グレースケール
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_bgr = img_rgb
    else:
        # PIL(RGB) -> OpenCV(BGR)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape

    # 結果初期値
    result = {
        "eye_size": 0.5, "face_length": 0.5, "jaw_roundness": 0.5,
        "brow_curve": 0.5, "contrast_level": 0.5, "warmth": 0.5
    }

    # -----------------------------------------------------
    # MediaPipeによる分析
    # -----------------------------------------------------
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        mp_results = face_mesh.process(img_rgb)

        if not mp_results.multi_face_landmarks:
            print(f"Warning: {image_path} から顔が検出されませんでした。デフォルト値を返します。", file=sys.stderr)
            return result

        lm = mp_results.multi_face_landmarks[0].landmark

        # =================================================================
        # 1. Contrast Level (彫りの深さ / ソース顔度)
        # =================================================================
        cx_min = int(lm[234].x * w) # 左頬
        cx_max = int(lm[454].x * w) # 右頬
        cy_min = int(lm[10].y * h)  # 眉間
        cy_max = int(lm[152].y * h) # 顎

        cx_min, cx_max = max(0, cx_min), min(w, cx_max)
        cy_min, cy_max = max(0, cy_min), min(h, cy_max)

        if cx_max > cx_min and cy_max > cy_min:
            face_roi = img_bgr[cy_min:cy_max, cx_min:cx_max]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            edge_score = np.mean(edge_magnitude)
            result["contrast_level"] = _normalize(edge_score, 20, 60)

        # =================================================================
        # 2. Warmth (雰囲気 / 笑顔度・優しさ)
        # =================================================================
        mouth_corner_avg_y = (lm[61].y + lm[291].y) / 2
        mouth_center_y = lm[0].y
        smile_val = (mouth_center_y - mouth_corner_avg_y) * h 
        
        eye_slant = (lm[33].y - lm[133].y) + (lm[263].y - lm[362].y) 
        eye_slant_val = eye_slant * h
        
        warmth_score = (smile_val * 0.7) + (eye_slant_val * 0.3)
        result["warmth"] = _normalize(warmth_score, -5.0, 10.0)

        # =================================================================
        # 3. その他の形状分析
        # =================================================================
        # Face Length
        face_height = _calc_dist(lm[10], lm[152], w, h)
        face_width = _calc_dist(lm[234], lm[454], w, h)
        if face_width > 0:
            aspect_ratio = face_height / face_width
            result["face_length"] = _normalize(aspect_ratio, 1.20, 1.55)

        # Eye Size
        left_eye_h = _calc_dist(lm[159], lm[145], w, h)
        right_eye_h = _calc_dist(lm[386], lm[374], w, h)
        avg_eye_h = (left_eye_h + right_eye_h) / 2
        eye_ratio = avg_eye_h / face_height if face_height > 0 else 0
        result["eye_size"] = _normalize(eye_ratio, 0.025, 0.055)

        # Jaw Roundness
        jaw_width = _calc_dist(lm[132], lm[361], w, h)
        chin_width = _calc_dist(lm[172], lm[397], w, h)
        if jaw_width > 0:
            jaw_ratio = chin_width / jaw_width
            result["jaw_roundness"] = _normalize(jaw_ratio, 0.35, 0.55)

        # Brow Curve
        brow_h_left = _dist_point_to_line(lm[105], lm[46], lm[70], w, h)
        brow_h_right = _dist_point_to_line(lm[334], lm[276], lm[300], w, h)
        avg_brow_h = (brow_h_left + brow_h_right) / 2
        brow_ratio = avg_brow_h / face_height if face_height > 0 else 0
        result["brow_curve"] = _normalize(brow_ratio, 0.02, 0.06)

    return result

# ---------------------------------------------------------
# メイン実行部
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. 画像フォルダの存在確認
    if not os.path.exists(PHOTO_DIR):
        print(f"Error: フォルダが見つかりません: {PHOTO_DIR}", file=sys.stderr)
        sys.exit(1)

    # 2. フォルダ内の画像ファイルをリストアップ
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(PHOTO_DIR) if f.lower().endswith(supported_exts)]

    if not files:
        print(f"Error: {PHOTO_DIR} に画像ファイルがありません。", file=sys.stderr)
        sys.exit(1)

    # 3. ランダムに1つ選ぶ
    chosen_file = random.choice(files)
    target_path = os.path.join(PHOTO_DIR, chosen_file)

    # ログ出力（どの画像が選ばれたか確認用）
    # JSON出力の邪魔にならないよう stderr に出すのがおすすめ
    print(f"Selected Image: {chosen_file}", file=sys.stderr)

    # 4. 解析実行
    result_data = analyze_image(target_path)

    # 5. 結果をJSONとして出力
    print(json.dumps(result_data, indent=2, ensure_ascii=False))