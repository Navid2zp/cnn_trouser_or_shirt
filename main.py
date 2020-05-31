from cnn.predict import predict
from cnn.data import create_data
from cnn.train import train

if __name__ == "__main__":
    action = input("Enter action (Predict, Create Data, Train): ")
    if action.lower() == "predict:":
        image_path = input("Enter the image path: ")
        prediction = predict(image_path)
        print(prediction)
    elif action.lower() == "create data":
        create_data()
    elif action.lower() == "train":
        train()
    else:
        print("unknown action")
