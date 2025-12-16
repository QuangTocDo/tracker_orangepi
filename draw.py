import cv2
import config # Import file config chính của dự án

def draw_info_panel(frame, bbox, display_attributes):
    """
    Vẽ bảng thông tin thuộc tính chi tiết cho một đối tượng.
    """
    if not display_attributes:
        return
        
    x1, y1, x2, y2 = map(int, bbox)
    info_x, info_y = x2 + 10, y1 + 15
    
    # Tính toán chiều cao panel
    panel_height = 20 # Padding ban đầu
    
    if 'gender' in display_attributes:
        panel_height += 30
    
    # >>> Thêm chiều cao cho Age & Race <<<
    panel_height += 40 # 2 dòng text (Age + Race), mỗi dòng ~20px
    
    panel_height += 20 
    brachium_colors = display_attributes.get('brachium_colors')
    if display_attributes.get('sleeve_type') and brachium_colors:
         panel_height += len(brachium_colors) * 20
         
    panel_height += 20
    thigh_colors = display_attributes.get('thigh_colors')
    if display_attributes.get('pants_type') and thigh_colors:
         panel_height += len(thigh_colors) * 20
         
    if display_attributes.get('skin_tone_bgr') is not None: 
        panel_height += 25

    # Vẽ nền panel
    cv2.rectangle(frame, (info_x, y1), (info_x + 250, y1 + panel_height), config.INFO_PANEL_BG, -1)
    
    # --- VẼ GENDER ---
    gender = display_attributes.get('gender')
    if gender:
        cv2.putText(frame, f"Gender: {gender.upper()}", (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, config.COLOR_INFO_TEXT, config.FONT_THICKNESS)
        info_y += 30
    else:
        cv2.putText(frame, "Gender: N/A", (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, config.COLOR_INFO_TEXT, config.FONT_THICKNESS)
        info_y += 30

    # --- VẼ AGE / RACE (AN TOÀN) ---
    age_race = display_attributes.get('age_race_analysis')
    print(f"Drawing Age/Race: {age_race}")
    if age_race and isinstance(age_race, dict):
        age_label = age_race.get('age', 'N/A')
        race_label = age_race.get('race', 'N/A')
        
        # Vẽ dòng Age
        cv2.putText(frame, f"Age: {age_label}", (info_x + 5, info_y), 
                    config.FONT, config.FONT_SCALE_INFO, (0, 255, 255), config.FONT_THICKNESS)
        info_y += 20
        
        # Vẽ dòng Race
        cv2.putText(frame, f"Race: {race_label}", (info_x + 5, info_y), 
                    config.FONT, config.FONT_SCALE_INFO, (255, 0, 255), config.FONT_THICKNESS)
        info_y += 20
    else:
        # Nếu không có dữ liệu Age/Race
        cv2.putText(frame, "Age: N/A", (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, (100, 100, 100), config.FONT_THICKNESS)
        info_y += 20
        cv2.putText(frame, "Race: N/A", (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, (100, 100, 100), config.FONT_THICKNESS)
        info_y += 20

    # --- HÀM VẼ MÀU SẮC ---
    def draw_color_details(part_name, part_type, colors):
        nonlocal info_y
        display_text = f"{part_name}: {part_type}" if part_type else f"{part_name}: Chưa xác định"
        cv2.putText(frame, display_text, (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, config.COLOR_CLOTHING_TEXT, config.FONT_THICKNESS)
        info_y += 20
        
        if part_type and colors:
            for color_info in colors:
                bgr = color_info.get('bgr')
                percentage = color_info.get('percentage', 0)
                text = f"- Color: {percentage:.1f}%"
                cv2.putText(frame, text, (info_x + 10, info_y), config.FONT, config.FONT_SCALE_INFO, config.COLOR_CLOTHING_TEXT, config.FONT_THICKNESS)
                if bgr is not None:
                    color_tuple = tuple(map(int, bgr))
                    cv2.rectangle(frame, (info_x + 200, info_y - 12), (info_x + 230, info_y + 5), color_tuple, -1)
                    cv2.rectangle(frame, (info_x + 200, info_y - 12), (info_x + 230, info_y + 5), (255,255,255), 1)
                info_y += 20
    
    draw_color_details("Sleeve", display_attributes.get('sleeve_type'), display_attributes.get('brachium_colors'))
    draw_color_details("Pants", display_attributes.get('pants_type'), display_attributes.get('thigh_colors'))

    # --- VẼ MÀU DA ---
    skin_bgr = display_attributes.get('skin_tone_bgr')
    if skin_bgr is not None and hasattr(skin_bgr, '__len__') and len(skin_bgr) == 3:
        cv2.putText(frame, "Skin Tone:", (info_x + 5, info_y), config.FONT, config.FONT_SCALE_INFO, config.COLOR_INFO_TEXT, config.FONT_THICKNESS)
        skin_color_tuple = tuple(map(int, skin_bgr))
        cv2.rectangle(frame, (info_x + 95, info_y - 12), (info_x + 125, info_y + 5), skin_color_tuple, -1)
        cv2.rectangle(frame, (info_x + 95, info_y - 12), (info_x + 125, info_y + 5), (255,255,255), 1)

def draw_tracked_objects(frame, tracked_objects):
    """
    Vẽ bounding box và thông tin cho các đối tượng được theo dõi.
    Đã cập nhật để lấy thông tin Age/Race từ final_attributes.
    """
    for track_id, obj_data in tracked_objects.items():
        bbox = obj_data.get('bbox')
        if not bbox:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        status = obj_data.get('status', 'pending')
        final_id = obj_data.get('final_id', f"Temp_{track_id}")
        
        # Xác định màu sắc bounding box
        if status in ['confirmed', 'identified']:
            color = (0, 255, 0) # Xanh lá
        elif status == 'tentative':
            color = (0, 165, 255) # Cam
        else:
            color = (255, 255, 0) # Xanh dương
        
        # Vẽ bounding box và label ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{final_id} [{status.upper()}]"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # --- LOGIC HIỂN THỊ MỚI ---
        final_attributes = obj_data.get('final_attributes')

        if final_attributes:
            display_data = {
                'gender': final_attributes.get('gender'),
                # >>> LẤY DỮ LIỆU AGE/RACE <<<
                'age_race_analysis': { 
                    'age': final_attributes.get('age'),
                    'race': final_attributes.get('race')
                },
                'sleeve_type': final_attributes.get('upper_type'),
                'pants_type': final_attributes.get('lower_type'),
                'skin_tone_bgr': final_attributes.get('skin_tone_bgr')
            }

            # Chuyển đổi định dạng màu
            upper_color_rgb = final_attributes.get('upper_color')
            if upper_color_rgb:
                display_data['brachium_colors'] = [{'bgr': upper_color_rgb[::-1], 'percentage': 100.0}]

            lower_color_rgb = final_attributes.get('lower_color')
            if lower_color_rgb:
                display_data['thigh_colors'] = [{'bgr': lower_color_rgb[::-1], 'percentage': 100.0}]

            draw_info_panel(frame, bbox, display_data)
        else:
            history_count = len(obj_data.get('history_attributes', []))
            analysis_text = f"Analyzing... ({history_count} samples)"
            cv2.putText(frame, analysis_text, (x1, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
    return frame