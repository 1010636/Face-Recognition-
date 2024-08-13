import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained FER-2013 model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    model = tf.keras.models.load_model('C:\\Users\\ANMOL YADAV\\Downloads\\emojis\\')  # Use your trained FER-2013 model
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Feedback messages
feedback_messages = {
    0: "You're looking happy! Keep smiling :)",
    1: "It seems like you're feeling sad. Remember, it's okay to feel this way. Talk to someone you trust.",
    2: "You seem angry. Try taking deep breaths and finding a way to relax.",
    3: "You're surprised! Something new and interesting might be happening.",
    4: "You look neutral. It's a good time to take a moment and reflect on your feelings.",
    5: "You seem disgusted. It might be helpful to take a break and focus on something positive.",
    6: "You appear fearful. Seek support if you need it, and take care of yourself."
}

def detect_face_and_mood(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        face_img = cv2.resize(roi_gray, (48, 48))
        face_img = face_img.astype('float32') / 255
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        
        try:
            prediction = model.predict(face_img)
            mood = np.argmax(prediction)
        except Exception as e:
            print(f"Error during prediction: {e}")
            mood = 6
        
        mood_to_emoji = {
            0: '😄',  # Happy
            1: '😢',  # Sad
            2: '😠',  # Angry
            3: '😮',  # Surprised
            4: '😐',  # Neutral
            5: '😏',  # Disgusted
            6: '😨'   # Fearful
        }
        emoji = mood_to_emoji.get(mood, '😐')
        feedback = feedback_messages.get(mood, "Stay positive!")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emoji, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display feedback
        cv2.putText(frame, feedback, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with video file path
    
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        frame = detect_face_and_mood(frame)
        
        cv2.imshow('Face and Mood Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

