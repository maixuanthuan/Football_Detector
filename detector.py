import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import argparse
import cv2
from tqdm import tqdm
from torchsummary import summary
def get_args():
    parser = argparse.ArgumentParser("Detector-Classifier pipeline")
    parser.add_argument("--video", "-v", type=str, default="data/football_test/Match_1864_1_0_subclip/Match_1864_1_0_subclip.mp4")
    parser.add_argument("--detector-checkpoint", "-d", type=str, default="player_detector.pt")
    parser.add_argument("--classifier-checkpoint", "-c", type=str, default="player_classification.pt")
    parser.add_argument("--output", "-o", type=str, default="test_output.mp4")
    args = parser.parse_args()
    return args

class PlayerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        del self.backbone.fc
        self.backbone.fc1 = nn.Linear(in_features=2048, out_features=3)
        self.backbone.fc2 = nn.Linear(in_features=2048, out_features=11)
        self.backbone.fc3 = nn.Linear(in_features=2048, out_features=2) #number of color

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.backbone.fc1(x)
        out2 = self.backbone.fc2(x)
        out3 = self.backbone.fc3(x)

        return out1, out2, out3


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(args.video)
    detector = torch.hub.load("yolov5", "custom","player_detector.pt", source="local")
    classifier = PlayerClassifier()
    checkpoint = torch.load(args.classifier_checkpoint)
    classifier.load_state_dict(checkpoint["state_dict"])
    detector.to(device).eval()
    classifier.to(device).eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(range(counter), colour='cyan')
    # while cap.isOpened():
    for idx in progress_bar:
        progress_bar.set_description("Frame: {}/{}\n".format(idx, counter))
        flag, original_frame = cap.read()
        if not flag:
            break
        frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        det_pred = detector(frame, size=1280)
        player_images = []
        for coord in det_pred.xyxy[0]:
            xmin, ymin, xmax, ymax, _, _ = coord
            player_image = frame[int(ymin):int(ymax), int(xmin):int(xmax), :]
            player_image = transform(player_image)
            player_images.append(player_image)
        player_images = torch.stack(player_images).to(device)

        with torch.no_grad():
            cls_pred = classifier(player_images)
        num_digits, unit_digit, team = cls_pred
        num_digits = torch.argmax(num_digits, dim=1)
        unit_digit = torch.argmax(unit_digit, dim=1)
        team = torch.argmax(team, dim=1)

        for (xmin, ymin, xmax, ymax, _, _), n, u, t in zip(det_pred.xyxy[0], num_digits, unit_digit, team):
            u = u.item()
            if n==2 or u==10:
                jersey_number = 0
            else:
                if n==0 :
                    if u==0:
                        jersey_number = 0
                    else:
                        jersey_number = u
                else:
                    jersey_number = u + 10
            if t==0:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            cv2.rectangle(original_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(original_frame, str(jersey_number), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2, cv2.LINE_AA)
        out.write(original_frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = get_args()
    inference(args)
    # classifier = PlayerClassifier()
    # random_input = torch.rand(10,3,224,224)
    # out1, out2 = classifier(random_input)