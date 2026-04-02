from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

model  = joblib.load('flower_model.joblib')
scaler = joblib.load('scaler.joblib')

SPECIES_INFO = {
    'Iris Setosa': {
        'description': 'A species of flowering plant in the genus Iris of the family Iridaceae. It belongs to the subgenus Limniris and the series Tripetalae. It is a rhizomatous perennial found in a wide range across and below the Arctic Circle, including in Alaska, Maine, Canada (including British Columbia, Newfoundland, Quebec and Yukon), Russia (including Siberia), China, Korea, and Japan. The plant has tall branching stems, mid green leaves and violet, purple-blue, violet-blue, blue, or lavender flowers, or, rarely, pink or white flowers.',
        'color': '#6aab63'
    },
    'Iris Versicolor': {
        'description': 'A species of Iris native to North America, in Eastern Canada and the Eastern United States. It is common in sedge meadows, marshes, and along streambanks and shores. The specific epithet versicolor means "variously coloured".',
        'color': '#a87d50'
    },
    'Iris Virginica': {
        'description': 'Large wetland iris from eastern North America. The largest of the three iris species, with broad, sweeping petals.',
        'color': '#7b6fa0'
    },
    'Rose': {
        'description': 'Rose, (genus Rosa), genus of some 100 species of perennial shrubs in the rose family (Rosaceae). Roses are native primarily to the temperate regions of the Northern Hemisphere. Many roses are cultivated for their beautiful flowers, which range in colour from white through various tones of yellow and pink to dark crimson and maroon, and most have a delightful fragrance, which varies according to the variety and to climatic conditions.',
        'color': '#b85555'
    },
    'Tulip': {
        'description': 'Tulip, (genus Tulipa), genus of about 100 species of bulbous herbs in the lily family, many of which are cultivated in temperate regions for their showy blooms. Native to Central Asia and Turkey, tulips are among the most popular of all garden flowers, and numerous cultivars and varieties have been developed.',
        'color': '#c47a3a'
    },
    'Sunflower': {
        'description': 'Sunflower, (genus Helianthus), genus of nearly 70 species of herbaceous plants of the aster family (Asteraceae). Sunflowers are native primarily to North and South America, and some species are cultivated as ornamentals for their spectacular size and flower heads and for their edible seeds. The Jerusalem artichoke (Helianthus tuberosus) is cultivated for its edible underground tubers.',
        'color': '#c8a820'
    }
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width  = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width  = float(data['petal_width'])

        features        = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        prediction    = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        classes       = model.classes_

        confidence = float(probabilities.max()) * 100
        all_probs  = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probabilities)}
        all_probs  = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            'success':       True,
            'species':       prediction,
            'confidence':    round(confidence, 1),
            'probabilities': all_probs,
            'info':          SPECIES_INFO.get(prediction, {'description': '', 'color': '#888'})
        })

    except (KeyError, ValueError) as e:
        return jsonify({'success': False, 'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
  import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
