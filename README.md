# VoiceLock
![Voice-Lock](https://github.com/Muhannad159/security-voice-code/assets/104541242/3036c85b-c28e-4303-af0f-b20d6df28c96)

## Project Overview

VoiceLock is a sophisticated software solution designed to provide secure access control based on voice recognition and fingerprint concepts. It offers two distinct operation modes: Security voice code and Security voice fingerprint. These modes cater to different security requirements and user preferences, ensuring flexibility and robustness in access control mechanisms.

## Usage and Importance

### Mode 1: Security Voice Code

In Security Voice Code mode, users can define a specific passcode sentence that serves as a key for access. This passcode sentence acts as a unique identifier, granting access only to individuals who accurately reproduce the predefined passphrase. This mode is particularly useful in scenarios where a standardized passphrase can be easily remembered and shared among authorized users, providing a convenient and secure method for access control.

### Mode 2: Security Voice Fingerprint

Security Voice Fingerprint mode offers a more personalized approach to access control by identifying individuals based on their unique voice characteristics. Users can specify which of the 8 saved individuals are granted access, allowing for precise and tailored access management. This mode is ideal for applications where individual authentication is paramount, such as in sensitive environments or high-security settings.

## Procedure
### Using a Random Forest ML model

The heart of VoiceLock app lies in its advanced voice and person detection capabilities, powered by a Random Forest model. Random Forest is a powerful machine learning algorithm that excels in classification tasks by constructing multiple decision trees and aggregating their predictions. This ensemble learning approach offers robustness and accuracy, making it well-suited for voice and person detection in security applications.

### Feature Extraction and Model Training

To train the Random Forest model, a comprehensive set of features is extracted from voice samples. These features include pitch, chroma, MFCC (Mel-frequency cepstral coefficients), and spectral contrast, among others. Each feature captures different aspects of the voice signal, providing rich information for classification. By leveraging a diverse range of features, the model gains a holistic understanding of voice characteristics, enhancing its ability to discriminate between individuals and passcode sentences.

### Euclidean Distance for Similarity Measurement

In addition to feature extraction and model training, Euclidean distance is employed to further improve accuracy. Euclidean distance is a simple yet effective metric for measuring the similarity between feature vectors. After extracting features from input samples and stored templates, the Euclidean distance between corresponding feature vectors is calculated. The model identifies the closest match based on the minimum distance, thereby increasing accuracy in voice and person recognition tasks.

## Tools Used

### Machine Learning

- **Random Forest**: Used for voice and person detection, providing robust classification capabilities.
- **Scikit-learn**: Python library utilized for implementing machine learning algorithms and model training.

### Feature Extraction

- **Librosa**: Audio analysis library used for extracting features such as pitch, chroma, and MFCC from voice samples.

### Similarity Measurement

- **NumPy**: Numerical computing library utilized for computing Euclidean distance between feature vectors, enabling similarity measurement and template matching.

## Conclusion

VoiceLock offers a comprehensive and versatile solution for access control, leveraging advanced machine learning techniques to ensure security and reliability. By combining innovative features, such as passcode-based access and voice fingerprinting, with state-of-the-art algorithms like Random Forest and Euclidean distance, the application sets a new standard in secure authentication systems.

## How to Use

1. **Clone this repository** to your local machine.
2. **Install the required dependencies** (`numpy`, `matplotlib`, `scipy`, `librosa`).
3. **Run the `main.py` file**.
4. **Use the application interface** to load a signal, adjust sampling frequency, add noise, and visualize the sampled and recovered signals.
5. **Experiment with different scenarios** and observe the effects on signal recovery.

## Dependencies

- Python 
- NumPy
- Matplotlib
- SciPy
- Librosa

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Muhannad159" target="_black">
    <img src="https://avatars.githubusercontent.com/u/104541242?v=4" width="150px;" alt="Muhannad Abdallah"/>
    <br />
    <sub><b>Muhannad Abdallah</b></sub></a>
    </td>
  <td align="center">
    <a href="https://github.com/AliBadran716" target="_black">
    <img src="https://avatars.githubusercontent.com/u/102072821?v=4" width="150px;" alt="Ali Badran"/>
    <br />
    <sub><b>Ali Badran</b></sub></a>
    </td>
     <td align="center">
    <a href="https://github.com/ahmedalii3" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110257687?v=4" width="150px;" alt="Ahmed Ali"/>
    <br />
    <sub><b>Ahmed Ali</b></sub></a>
    </td>
<td align="center">
    <a href="https://github.com/hassanowis" target="_black">
    <img src="https://avatars.githubusercontent.com/u/102428122?v=4" width="150px;" alt="Hassan Hussein"/>
    <br />
    <sub><b>Hassan Hussein</b></sub></a>
    </td>
      </tr>
 </table>



