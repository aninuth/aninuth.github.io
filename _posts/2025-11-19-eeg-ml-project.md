---
title: "7 Terabytes of Humility"
date: 2025-08-26
permalink: /posts/2025/08/eeg-ml-project/
tags:
  - Self Supervised Learning
  - EEG
  - neuroscience
  - research
  - signal processing
  - GNNs
  - Autoencoders
---

Not every research project ends with a publication, but that doesn't mean it wasn't worth doing. This post chronicles my most meaningful research project yet - I learned more about myself and the research process during this endeavor than I did during the rest of my academic career. I'll be highlighting things I learned using the 💡 symbol - check it out if you want my insights - I promise it's better than learning them the hard way.

## The Problem Space

Intracranial Electroencephalography (iEEG) data is notoriously noisy, high-dimensional, and has incredibly high temporal resolution. It is also incredibly challenging to find labeled data, making it a ripe use case for Self Supervised Learning (SSL). After discussions with *Professor Yoga Varatharajah*, we found quite an interesting gap in the existing literature - no one had yet managed to identify a relationship between iEEG spectral features and brain function. 

GNNs are well-suited to handle the spatial and temporal dimensions of iEEG data, but much of the literature in this space focused on seizure detection and sleep staging. Much of the pioneering work in this space was done by Dr. Varatharajah, to be sure, but this analysis frequently involved examining the EEG data to identify things like pre-ictal activity and seizure onset/duration. There wasn't a whole lot of work done on getting a more holistic understanding of brain function/capacity from resting-state iEEG data.

Together, we started a collaborative project with *Dr. David Darrow*, using a novel dataset that his lab had collected from epilepsy patients that were soon undergoing surgery. In addition to 24 hours of continuous iEEG recordings, we also had a battery of results from neuropsychological tests that the patients had completed.

The clinical motivation was clear: If we could extract any meaningful insights from the iEEG data, we could develop a much clearer understanding of the progress of a patient's brain health - since many of these tests are not repeatable, having a continuous measure of brain health would be invaluable. This would also make closed loop stimulation much more effective, as we would have the ability to monitor a patient's brain health in real-time and adjust stimulation parameters accordingly.

## Breaking Down the Problem

I started out by collecting the 7 TB of data (on a surprisingly compact hard drive) from a post-doc at 6 in the morning. Walking around campus with that was certainly trippy. Then, I just opened each file in VSCode. Just kidding!

When you're staring at raw EEG data for the first time, it's overwhelming. For those of you who have experience with signal processing, this is what the Power Spectral Density (PSD) looks like for a single patient: 

![PSD](/assets/images/psd.png)



### 1. The Data Challenges 
1. There's way less literature (and data) on intracranial EEG data than there is for scalp EEG data. 
2. Most healthy people aren't volunteering to have their skulls drilled into in order to have electrodes implanted. This means that there's a dearth of healthy control subjects.
3. Intracranial EEG implants are far less standardized compared to scalp EEG electrodes. With scalp EEG caps, it is common for electrodes to be placed in a standard configurations, but this is not the case for intracranial EEG implants. Each patient's electrodes are placed based on their specific anatomy, and the electrodes are often not even in the same locations for different patients.
4. The data has both spatial and temporal dimensions, meaning that the traditional methods of time-series analysis and spatial analysis do not suffice.

### 2. Preprocessing Pipeline
Raw EEG is messy. I had to make informed decisions about:
- **Artifact removal**: Eye blinks, muscle movements, and electrical noise all contaminate the signal.
- **Filtering approach**: What frequency bands matter for my specific question? 
- **Segmentation strategy**: How long should each sample window be?
- **Outlier Thresholds and removal**: How aggressive should I be when defining outliers? Should I remove individual epochs or entire channels? 

<details>
<summary>💡 <b>"Good-enough Decision Making"</b></summary>
<br>
I learned this the hard way: I initially spent over a week trying to find the "optimal" preprocessing pipeline, exploring every possible parameter combination. Each of these decisions could have been a series of papers in their own right, and I treated them that way. What I should have done was make informed "good-enough" decisions early based on literature-supported baselines, then iterated. Every preprocessing choice was reversible, but I stayed stuck in analysis paralysis writing exploratory code when I could have been learning from actual model results. Starting with reasonable defaults and iterating beats perfectionism.
</details>

### 3. Setup
The most central design decisions are here: 
**Feature Engineering**: Should I hand-engineer features (spectral power, connectivity measures, etc.) or use end-to-end deep learning?
**Electrode Parcellization**: Should I use a priori parcellization (based on brain region definitions, etc) or use a data-driven approach based on test scores and electrode distribution?
**Fully Connected versus sparse network**: How important is message passing between distant nodes?
**Evaluating Embeddings**: How should I perform dimensionality reduction on the embeddings and correlate them to test scores in a way that is patient-agnostic?

## The Research Process 

<details>
<summary>💡 <b>"Starting Out Small - Sanity Checks First"</b></summary>
<br>
Being totally honest - I didn't start with the simple experiments below. Eager to write complex code, I spent the next few weeks building complex, heavy GNNs and VAEs, convinced that different architectures would yield promising results. I could fill a book with the awful loss curves I generated, and it wasn't until I had pulled a good chunk of my hair out did I realize that I should have started with some simple sanity checks and smaller-scale experiments.
</details>

Working with a Variational Autoencoder (VAE) using the graph structure I had established, I was able to achieve pretty impressive results... under specific circumstances. The model learned and generalized incredibly well when working with data from a single patient, and was almost as impressive under a within-patient train/test split. However, the results did not generalize *at all* to held-out patients. This led me to do some digging, leading to my first major finding:

### Diagnosis 1: Node Overlap
I dug into the data structure to understand why the graph model couldn't generalize from Patient A to Patient B. I wrote a script to map the 3D coordinates of every electrode across the patient cohort into a standardized grid. The results were stark:
At a 10x10x10 resolution, the overlap between patients was 99%.
At a 100x100x100 resolution, the overlap dropped to ~25%.
Beyond that, there was almost no overlap at all.

This meant I was stuck between a rock and a hard place. The difference in electrode locations was extreme, but I either had to use large grid cells (losing patient-to-patient spatial differences) or be comfortable using smaller cells, with less learning power from patient-to-patient. I decided to start with 100x100x100, providing me with at least some overlap that could be useful for generalization. Very slowly, though. Way slower than I was expecting. Leading me to: 

### Diagnosis 2: Graph Structure And Definition
I was working with a grid of 100x100x100 cells (merging all electrodes within each cell). This meant that I had 1M nodes, and $(10^6 \times 10^6) / 2 = 10^{12}/2$ edges.

However, only 5162 cells were actually active at all (had electrodes in them for at least one patient). 5162 nodes means ~$10^7$ edges, a signficant reduction. The downside: If I encountered a new patient from outside the dataset, they might have electrodes in cells that were not included, making it tough to generalize. 

Once I made this change, training and testing was much faster. This allowed me to test different combinations of features, and I ended up initially settling on the following: 

Node feature: Delta/Beta Ratio
Edge feature: Spectral Coherence

This initial setup was starting to look a little better, but I still wasn't getting anything near what I was hoping for. 

### Diagnosis 3: Other People Exist Too
<details>
<summary>💡 <b>"Talk To Other People"</b></summary>
<br>
Make sure you're accounting for other people's actions. Have they pre-processed your data already? Are there limitations to their data capture that you need to be aware of? The old programming joke (10 hours of googling can save you 10 minutes of reading the manual) is true for research too, except sometimes those 10 minutes come from writing an email.
</details>


I reached out to the postdoc with a question about the data. In the ensuing discussion, he revealed to me that the data I was provided was actually pre-processed already, and the delta band might have been significantly impacted. So much for the clinical relevance of the delta/beta ratio (the only node feature I was using in my prototype).

Now that I had a smaller graph, I was able to try a few different node features. Rather than providing power band values, I simply provided the spectrogram of each node for the duration of the epoch. I also modified my pre-processing technique to be less reliant on powerband filtering.

Playing around with different electrode splits here led me to my next finding:

### Diagnosis 4: Electrode Types Matter
I overlooked a major difference between scalp and intracranial EEG, and it cost me dearly. It wasn't until a few months had passed that I realized I was working with a combination of grid, strip, and depth electrodes. This difference resulted in different power band spectra, and each one represented brain function slightly differently. By attempting to aggregate different types of electrodes together (or get my model to generalize from one to the other), I was muddying the waters. 

Separating by electrode type helped quite a bit, and I could probably have come up with a more complex electrode-type feature that would have helped even more. As it stood, the experimental sequence I had at the moment was as follows:

Masking nodes/edges, reconstruct raw signals/spectra -> VAE to generate node/graph level representation.

////spectrogram loss

## Bringing It Back: Brain Health

I had a model that was somewhat competent at learning and generalizing across patients. Now for the fun part - tying it into brain health. Once I had embeddings for each individual epoch, I attempted to perform dimensionality reduction on the embeddings and correlate them to test scores in a way that was patient-agnostic. This worked... but it wasn't particularly great. Barely statistically significant, about 40% accuracy compared to a 20% baseline. Even though the embeddings were meaningful, there just wasn't enough data in them that was useful for predicting the test scores. 

## Future Directions

At various points in this process, I stumbled upon things that could have been dissertations all on their own. There are a few that are interesting to me, and I'm listing them below in case anyone is inspired (or I continue working on this in the future). 

### Comparing different parcellization schemes
- The dataset provided used the Yale atlas for electrode-region identification, but I wonder if building a novel parcellization scheme based on test scores would be helpful. 
- Some combination of a grid/cell based approach to maintain even density, while considering brain regions and their associated functions, could make projects like this easier in the future.

### Different ways to evaluate brain embeddings
- Perhaps the test scores themselves are pretty noisy - there might be a better way of evaluating the quality of brain embeddings.
- Science Fiction has long dreamed of a "connectome" or a "brain fingerprint" - would it be possible to use these embeddings to identify individual patients? (For what it's worth, I was able to do that pretty easily using this dataset, but it turned out the model was over-indexing on electrode placement. Womp womp.)

### Different SSL tasks/approaches
- Would the embeddings be any better if different SSL tasks were used? Would adversarial graph auto-encoders make a difference? What about attention-based models? After all, they do claim it's all you need.
- Would adding spatial objectives to training help? Currently, we treated all nodes as the same, even though the power band is noticeably different throughout the brain. By adding some sort of spatial objective (predict where a node is located, based on power band?) we could probably improve SSL.
- Comparing different SSL techniques could make for an interesting paper, assuming one is head-and-shoulders above the rest.

### Creating a Normative (healthy) IEEG map by using multiple unhealthy patients
- Can we combine the data we have to generate a map of what healthy IEEG data might look like? Stitching together data from different patients, using regions in which their data is as close to healthy as we can get? 
- Reminder that healthy IEEG data basically does not exist, so this would be a major milestone.



## What I Learned

This project didn't result in a publication, but it gave me something arguably more valuable:

1. **The research process is iterative**: The sooner you learn this, the more success you'll have.

2. **"Good enough" is often the right choice**: Waiting for the perfect preprocessing pipeline would have meant never starting. I made informed decisions and iterated.

3. **Domain knowledge matters**: Understanding EEG physiology helped me debug why certain approaches failed.

4. **Negative results are results**: Just because I didn't achieve state-of-the-art performance doesn't mean the work had no value.

5. **Research is more than results**: Experimental design, critical thinking, and understanding limitations are as important as the final numbers.

## Technical Details

For those interested in the specifics:

**Framework**: PyTorch-Geometric
**Key libraries**: MNE-Python

For additional details, feel free to contact me!
---

*This project reinforced something important: research capability isn't just about getting publishable results. It's about asking good questions, designing rigorous experiments, making informed decisions under uncertainty, and learning from both successes and failures.*
