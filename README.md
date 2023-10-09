# Emolit Train

## Instructions
Extract the data into the `data` dir so it looks like this: `data/emolit`.
Or for the multilingual data: `data/emolit_multilingual`

Install the requirements in `requirements.txt` (consider virtualenv).

Change any parameters in the `soft_train.py` file (e.g. encoder model, batch
size, number of epochs, ...).

Run: `python soft_train.py`. This should train a model and save it to the `model` directory.
Run: `python multilingual_train_eval.py`. This should train a model and save it to the `model/multilingual` directory.

## Description
Literature sentences from [Project Gutenberg](https://www.gutenberg.org/). 38 emotion labels (+neutral examples). Semi-Supervised dataset. 

## Article
[Detecting Fine-Grained Emotions in Literature](https://www.mdpi.com/2076-3417/13/13/7502)

Please cite:
```plain text
@Article{app13137502,
AUTHOR = {Rei, Luis and Mladenić, Dunja},
TITLE = {Detecting Fine-Grained Emotions in Literature},
JOURNAL = {Applied Sciences},
VOLUME = {13},
YEAR = {2023},
NUMBER = {13},
ARTICLE-NUMBER = {7502},
URL = {https://www.mdpi.com/2076-3417/13/13/7502},
ISSN = {2076-3417},
DOI = {10.3390/app13137502}
}
```
## Abstract
Emotion detection in text is a fundamental aspect of affective computing and is closely linked to natural language processing. Its applications span various domains, from interactive chatbots to marketing and customer service. This research specifically focuses on its significance in literature analysis and understanding. To facilitate this, we present a novel approach that involves creating a multi-label fine-grained emotion detection dataset, derived from literary sources. Our methodology employs a simple yet effective semi-supervised technique. We leverage textual entailment classification to perform emotion-specific weak-labeling, selecting examples with the highest and lowest scores from a large corpus. Utilizing these emotion-specific datasets, we train binary pseudo-labeling classifiers for each individual emotion. By applying this process to the selected examples, we construct a multi-label dataset. Using this dataset, we train models and evaluate their performance within a traditional supervised setting. Our model achieves an F1 score of 0.59 on our labeled gold set, showcasing its ability to effectively detect fine-grained emotions. Furthermore, we conduct evaluations of the model's performance in zero- and few-shot transfer scenarios using benchmark datasets. Notably, our results indicate that the knowledge learned from our dataset exhibits transferability across diverse data domains, demonstrating its potential for broader applications beyond emotion detection in literature. Our contribution thus includes a multi-label fine-grained emotion detection dataset built from literature, the semi-supervised approach used to create it, as well as the models trained on it. This work provides a solid foundation for advancing emotion detection techniques and their utilization in various scenarios, especially within the cultural heritage analysis.


## Labels
    - admiration: finds something admirable, impressive or worthy of respect
    - amusement: finds something funny, entertaining or amusing
    - anger: is angry, furious, or strongly displeased; displays ire, rage, or wrath
    - annoyance: is annoyed or irritated
    - approval: expresses a favorable opinion, approves, endorses or agrees with something or someone
    - boredom: feels bored, uninterested, monotony, tedium
    - calmness: is calm, serene, free from agitation or disturbance, experiences emotional tranquility
    - caring: cares about the well-being of someone else, feels sympathy, compassion, affectionate concern towards someone, displays kindness or generosity
    - courage: feels courage or the ability to do something that frightens one, displays fearlessness or bravery
    - curiosity: is interested, curious, or has strong desire to learn something
    - desire: has a desire or ambition, wants something, wishes for something to happen
    - despair: feels despair, helpless, powerless, loss or absence of hope, desperation, despondency
    - disappointment: feels sadness or displeasure caused by the non-fulfillment of hopes or expectations, being or let down, expresses regret due to the unfavorable outcome of a decision
    - disapproval: expresses an unfavorable opinion, disagrees or disapproves of something or someone
    - disgust: feels disgust, revulsion, finds something or someone unpleasant, offensive or hateful
    - doubt: has doubt or is uncertain about something, bewildered, confused, or shows lack of understanding
    - embarrassment: feels embarrassed, awkward, self-conscious, shame, or humiliation
    - envy: is covetous, feels envy or jealousy; begrudges or resents someone for their achievements, possessions, or qualities
    - excitement: feels excitement or great enthusiasm and eagerness
    - faith: expresses religious faith, has a strong belief in the doctrines of a religion, or trust in god
    - fear: is afraid or scared due to a threat, danger, or harm
    - frustration: feels frustrated: upset or annoyed because of inability to change or achieve something
    - gratitude: is thankful or grateful for something
    - greed: is greedy, rapacious, avaricious, or has selfish desire to acquire or possess more than what one needs
    - grief: feels grief or intense sorrow, or grieves for someone who has died
    - guilt: feels guilt, remorse, or regret to have committed wrong or failed in an obligation
    - indifference: is uncaring, unsympathetic, uncharitable, or callous, shows indifference, lack of concern, coldness towards someone
    - joy: is happy, feels joy, great pleasure, elation, satisfaction, contentment, or delight
    - love: feels love, strong affection, passion, or deep romantic attachment for someone
    - nervousness: feels nervous, anxious, worried, uneasy, apprehensive, stressed, troubled or tense
    - nostalgia: feels nostalgia, longing or wistful affection for the past, something lost, or for a period in one’s life, feels homesickness, a longing for one’s home, city, or country while being away; longing for a familiar place
    - optimism: feels optimism or hope, is hopeful or confident about the future, that something good may happen, or the success of something
    - pain: feels physical pain or is experiences physical suffering
    - pride: is proud, feels pride from one’s own achievements, self-fulfillment, or from the achievements of those with whom one is closely associated, or from qualities or possessions that are widely admired
    - relief: feels relaxed, relief from tension or anxiety
    - sadness: feels sadness, sorrow, unhappiness, depression, dejection
    - surprise: is surprised, astonished or shocked by something unexpected
    - trust: trusts or has confidence in someone, or believes that someone is good, honest, or reliable

## Dataset
[EmoLit (Zenodo)](https://zenodo.org/record/7883954)
[EmoLit Translated (Zenodo)](https://zenodo.org/record/8420877)

## Code
[EmoLit Train (Github)](https://github.com/lrei/emolit_train)

## Models
  - [LARGE](https://huggingface.co/lrei/roberta-large-emolit) 
  - [BASE](https://huggingface.co/lrei/roberta-base-emolit) 
  - [DISTILL](https://huggingface.co/lrei/distilroberta-base-emolit)
  - [Multilingual](https://huggingface.co/lrei/xlm-roberta-base-emolit-multilingual): en, nl, fr, it

