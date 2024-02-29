import torch

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.version.cuda)

'''

        # บันทึกรูปภาพแต่ละ object # เขียนเพิ่ม
        for i, identity in enumerate(identities):# เขียนเพิ่ม
            obj_id = int(identity)# เขียนเพิ่ม
            x1, y1, x2, y2 = map(int, bbox_xyxy[i])# เขียนเพิ่ม
            crop_img = im0[y1:y2, x1:x2]# เขียนเพิ่ม

            # บันทึกรูปภาพโดยใช้ ID เป็นชื่อไฟล์
            cv2.imwrite(str(self.save_dir / f"{obj_id}.jpg"), crop_img)# เขียนเพิ่ม

'''

'''
for i, identity in enumerate(identities):
    obj_id = int(identity)
    x1, y1, x2, y2 = map(int, bbox_xyxy[i])
    crop_img = im0[y1:y2, x1:x2]

    cv2.imwrite(str(self.save_dir / f"{obj_id}.jpg"), crop_img)

'''