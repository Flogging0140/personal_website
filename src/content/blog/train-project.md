---
title: "The Train Project"
description: "Using AI to detect the train in a rural town. Why? Because..."
date: 2026-09-06
---

![image maybe](/blog/train-project/train-header-16:9.jpg)

# What is this Project?! 🤔

This is not intended to be an academic writeup or overly verbose. Instead my intention is to professionally explain the project, decisions made, etc; if anyone is curious they may read.

The purpose of this project has been to detect when and measure how long the train blocks the tracks into a town called Swift Current, SK. Collecting this data is not out of malice. It simply has been a curiosity of mine for years.

> George Mallory: climbs Everest - "Because it is there".

> Me: tracking the train w/AI - "Because it was blocking the tracks Monday morning".

<!-- 
TODO:

- Intenet Issues. 
  - Hot potato.
  - Minimizing network usage.
  - ADB connection to phone.
- Model Design.
- Fun Exceptions & Fixes
  - Rounding w/int that cut-off values. 
- Ansible
- The Data (talking about the data).

-->

# Index
* [Labelling](#labelling-the-data).
* [Generating the Data](#generating-training-data).
* [System Architecture](#system-architecture).
* [Training the AI](#training-the-ai-model).
* [Credits](#credits).

---

# TLDR

1. Listen to 804+ hours of train sounds, then write start & stop times of trains. 
2. Turn the sound into lists of lists of numbers, then use that to train an AI model. 
3. Use the trained AI model, a mini-computer, and mic, to predict the train. 

Step z: *Solve hundreds of obscure technical issues... Done!*

---

# Labelling The Data

![](/blog/train-project/labelling.gif)

## Summary

To label the training data I used TenacityAudio, which allowed me to add labels with corresponding timestamps correlating to .wav files.

---

#### Where is the Audio from?

One may ask, "*where did the audio come from*", and reasonably so. On this, the hardware records audio of a train from close enough distance it can be heard, with some exceptions to quality of sound, which I will not get into.

During the initial phase data was recorded at intervals of the day. From these recordings there are mainly two or three events to care about. These are train, bell, and no train.

The arduous process of labelling involved opening each audio file in Tenacity, labelling when the train arrived, when mostly the bell could be heard, and when the train was not there. At the time of writing I have manually labeled a cumulative 804 hours of audio.

<!-- image2 -->

## Limitations to Consider

There were limitations to this type of labelling and here is how I dealt with them. Firstly, limitations. Drawing consistent boundaries, overlapping events, and direction.

#### Labelling Boundaries

Drawing consistent boundaries has the potential to be difficult when events don't happen in a single moment but gradually. Sometimes the train arrives fast, other times slowly. The question may be raised, if the train arrives slowly, at what moment is it labelled train. I picked the most repeatable rule I could follow, if I hear the train at any volume without straining, label it train.

#### Event Overlap

Both bell and train may, and often do, overlap. Well, how is labelling done when there are multiple events overlapping. While there were multiple ways to approach this, I took what I considered to be the simplest because I knew how I wanted the system to work in the big picture. Because I want to measure when the tracks are blocked and either train or bell triggers the crossing arms, the overlap is less relevant. If the train stopped on the tracks I could hear mostly the bell, so labelled the bell, otherwise labelled the train.

Direction is part of the situation as well. If the train comes from side A, the bells trigger first, if it comes from side B, the train is heard first. Not much was needed to be done because the model's confidence values grew as the train got gradually closer.

#### Gaps in the Data

This is the reason the [window functionality](#window-functionality) feature exists. Because in the data, bells are sometimes very audible while other times less audible. Since the location changes I am assuming there are forces of nature acting on the sound as it moves, which impacts the training data. 

*Maybe you will have to zoom in (see below). I have confirmed the bell is present the entire duration and sometimes it is very easy to hear while others times not so.* 

![](/blog/train-project/bell_less_gaps.png)
![](/blog/train-project/bell_more_gaps.png)

NOTE: *if you the reader, have experience with this type of data, send me a direct message with your opinion on how you would have interpreted it.*

# Generating Training Data

![](/blog/train-project/training_data_generation.png)

## Cleaning Data

![](/blog/train-project/train_folder_size.png)

Most of the data cleaning I have done has been in tabular datasets where there are missing values or outliers. Here we have no missing values (*if we do, they are dropped*) and most outliers are addressed when labelling.

---

## Splitting Data

Splitting data has seen a couple iterations.

When samples (*i.e. each 2 second clip*) was still being saved as individual `.png` files, they were split between Train, Validation, and Test groups. Samples were switched to be saved as `.npz` files (*not pictures, compressed number lists*). 

At the latter point I noticed a sample from one file, may be included in test, while another sample from that same file may be included in either validation or test. This appeared to be data leakage because those two samples are NOT independent due to conditions at the time of recording and that specific train. The concern is the model may be able to predict it because it can recognise it from memory as some previous train, not generally a train. Theh solution was to use `GroupShuffleSplit`, keeping the related samples, 2-second samples per-file, in the same group.

---

## Processing Data

![](/public/blog/train-project/power_vs_db_distribution.png)

Power values from the log mel spectrogram were converted to decibel values because of the differences in scaling. The scale and distribution differ as power values scale linearly, and values can be spaced far apart in a large range. Decibel values scale logarithmically, are closer to human hearing, and values are distributed in a smaller range. 

To normalize the data, min-max normalization was used. This allows a better comparrison of data since they are now in the same range, based on a 10% sample of min & max values to represent the population. Since this sometimes results in values that are slightly outside the range, those values are clipped to fit, which for this case has a negligible affect on training data. 

$$
x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

$$
S_{\text{norm}} = \frac{S_{db} - g_{\min}}{g_{\max} - g_{\min}}
$$

After data is processed it is available in sub directories of the generated data directory, where sub directories are for train, validation, and test respectively, where each has a csv files for corresponding labels & tensor indices.

---

# System Architecture

TODO: BIG PIC HERE...

## Summary

The system started simply with a Raspberry Pi (*Pi*) where the intention was to send consistent loudness to a database, which would be queried to explain the data later. It then evolved into a much larger system of interacting parts, each with important roles to play.

To preface this, yes it could have been done simpler but thinking about it now, this is the right level of simplicity.

"An idiot admires complexity, a genius admires simplicity" — Terry A. Davis.

---

## Inference Loop (*main loop*)

![](/blog/train-project/main_inference_loop_v4.png)

The main driving function in the application is this loop in `pi_inference.py`. This sits on the Raspberry Pi 4b.

```python
# open a train event if not already open
if len(window) == WINDOW_SIZE and hits_train >= THRESHOLD_HITS_TRAIN:
    if not currently_train:
        subproc_trainrec2 = open_train(subproc_trainrec2)
        currently_train = True

# close a train event if already open and confirmed no train
if (
    len(window_no_train) == WINDOW_SIZE
    and hits_no_train >= THRESHOLD_HITS_NO_TRAIN
):
    if currently_train:
        subproc_trainrec2 = close_train(subproc_trainrec2)
        currently_train = False
```

### Turning up Sampling

To start the system was taking 2 second samples every 5 seconds. Because of the 5 second wait this made the [window functionality](#window-functionality) take longer to assist. Additionally if that 2 second clip was taken at a moment where the sound was more abnormal it could give a lower confidence value even if the 2 seconds before were normal. 

To address this, the simple solution was to read from audio continuously and save 2 second clips into a queue, that is then read by the main inference loop. 

Of course that was easier said than done. It required adding the `ContinuousReaderHelper` to manage a thread-safe queue, reading and writing to a buffer with locks, and draining exactly `2.0` seconds of bytes from the buffer, then in the main inference loop getting the earliest-added item from that queue. 

### Fun Error: time.sleep vs counting bytes

Once upon a time I was using `time.sleep(2)` and did seriously believe that blocking the thread for two seconds would result in a 2 second recording. Some days later I looked at it and though: "*hmm, this is not okay*". The reason it was not okay is because when small operations are executed they take time, reading from the mic even takes time to start. Since I didn't want to start and stop 2 second recordings, easier in some processes than others, I decided on counting the number of bits that was equal to exactly `2.0` seconds. 

From Google, I used this formula:
<!-- remember, I used KaTeX, not MathJax this time -->
$$
\text{bytes} = \text{sample\_rate} \times \text{channels} \times \frac{\text{bits\_per\_sample}}{8} \times \text{seconds}
$$

In code: 
```python
BYTES_PER_2_SEC = int(self.device_sr * 1 * (16 / 8) * 2)
```

With this, I was able to get exactly two seconds of bytes, read from a buffer. This is what gets given to the model each time. Why this matters is because the model uses `nn.AdaptiveAvgPool2d`. I could give it more or less than 2.0 seconds, however, since it was trained on exactly two seconds and because this layer would use averaging that may make the data look differently than in training, I've decided to give exactly 2.0 seconds. 


## AI Predictions
![](/blog/train-project/predict_function_compact.png)

Before we continue, <u>I want you to understand where the confidence values come from</u>; they are numbers like `.99, .05, etc.` from "the ai". I will not get too into the weeds yet on the model itself.

Simply, we give a tensor (*lists of lists of numbers*) to the model (*the "ai"*), which then guesses the class label. At time of writing the class labels are `["train", "bell", "no_train"]`. When I refer to a train event, that may either be **train** or **bell**, while **no_train** is the probability of neither posative events; the relevance of this specifically is when I discuss the [windows](#window-functionality) for train & no_train repectively. 

## Window Functionality
![](/blog/train-project/window_functionality_flowchart.png)

### The Challenge
One of the challenges in this project has been variance (correct word?) in 2 second clips. Meaning the sound of the train both changes over time which shows differently depending where it starts in the timespan of 2 seconds and influences how visible the bell is during those 2 second clips. 

### The Solution

Rather than spending large amounts of time on what I would deem to be a much harder problem, I circumvented the variance in 2-second clips by implimenting a window threshold system. 

There is a window for trains and no trains (*yes, I probably could just use only one window*). Both store confidence values (`i.e. [.55, .65. .99, ...]`), which are queued each inference iteration (*a big loop*). Again, each iteration, the number of confidence values in each window is checked to see if the sum of values over the minimum is greater than n where n is a pre-set threshold hits. 

<!-- indicator fn w/Iverson bracket notation -->
<!-- indicator fn: a block ouputs 1 or 0 -->
<!-- Iverson Bracket Notation: square brackets, named after some guy -->
$$
\sum_{i=1}^{N} \mathbb{1}[\,c_i > \tau\,] > n
$$

| | |
|---|---|
| $c_i$ | a confidence value in the window |
| $N$ | how many values are currently in it |
| $\tau$ (tau) | the minimum confidence threshold |
| $n$ | the pre-set threshold hits |

### Errors, Notifications, & Flask
![](/blog/train-project/pi_inference_outbound_tailscale.png)

Way earlier in the project I wanted to have notifications for when specific events happened. At that time I had added [ntfy](https://ntfy.sh/) so I could be notified for specific events. It was very useful when there was an error written to a log file that I wanted to immediatley be notified of. 

More recently I have started using [GlitchTip](https://glitchtip.com/) because it is more visually appealing and better suited to tracking errors in different locations i.e. phone, pi, server laptop, dev laptop, cloud server, etc. I want to briefly say I will be using this software on future projects that have multiple parts where non-blocking errors can occur in production environments. 

![](/public/blog/train-project/glitchtip_stock.webp)

Flask is running in [Termux](https://termux.dev/en/) on a Google Pixel (*more on this later*). At this point, just understand that open, close, & heartbeat events are posted to Flask, running on a phone, by way of a USB cable, for internet reasons...

## Device Interaction

![](/blog/train-project/device_map_overview_v3.png)

The entire system involves multiple devices and entities. Those devices are a Raspberry Pi 4b (*computer*), a Google Pixel (*phone*), a RØDE microphone, a server running on a laptop, a cloud server in New York, an external hard drive, and cables.

*Each device has a specific purpose, see below.*

### Raspberry Pi 4b

<img src="/blog/train-project/folder_pi.png" alt="" class="img-float-right" />
<!-- ![](/blog/train-project/folder_pi.png) -->

The Pi runs on unreliable wifi but does not have a strict limit. It runs the AI / ML inference on the edge. A USB cable connects it to the Google Pixel phone. ADB forward allows posting from the Pi to the Flask server on the phone. 

A difficulty of using the phone has been accessing it without using ssh. While `adb shell` can be used to remote  into the Pi over the `USB` cable, python, [tmux](https://github.com/tmux/tmux/wiki), and the requirements.txt file have been installed in [Termux](https://termux.dev/en/) so are only available there. It is possible to switch to the termux user by finding the running application, however, this comes with another caveat. The caveat being that session does not have internet access. Additionally files transferred had to be transferred from the entry point of `adb shell` to a directory Termux has access to and their permissions modified. One saving grace of this setup is starting a tmux session in person, which has internet. 

### Google Pixel

<img src="/blog/train-project/folder_phone.png" alt="" class="img-float-right" />
<!-- ![](/blog/train-project/folder_phone.png) -->

The phone is running a custom internet data plan of 300 megabytes, which renews when out or in 10yrs. Flask receives posted events from the pi for heartbeats, train events opening, and train events closing. `MQTT` is used to "*publish*" messages to the broker with the reliable but limited internet.

### Server Laptop

<img src="/blog/train-project/folder_server_laptop.png" alt="" class="img-float-right" />
<!-- ![](/blog/train-project/folder_server_laptop.png) -->

The server laptop, running on [TailScale](https://tailscale.com/), subscribes to [Mosquitto](https://mosquitto.org/) for events. [GlitchTip](https://glitchtip.com/) is run on this device, and accessible for those on the Tailnet. In a recept update it now can have exceptions sent to it from devices not on the Tailnet. Messages from MQTT are sent to the database.

### Developer PC

<img src="/blog/train-project/folder_model_training.png" alt="" class="img-float-right" />
<!-- ![](/blog/train-project/folder_model_training.png) -->

The developer pc is responsible for both generating the training data and training a model on that data. 

[MLflow](https://mlflow.org/) is used on this device to monitor model training progress. 

[Ansible](https://docs.ansible.com/) is run from this device to automate deployment to other devices for updating scripts (*which is quite a process*). 

### Server in New York

Hosts the [PostgreSQL](https://www.postgresql.org/) database and [Eclipse Mosquitto](https://mosquitto.org/) (MQTT Broker). <u>TODO:</u> Paxton can include content on the other script and the app if he wants. 

---
<!-- TODO: see Google Doc for internet issues start. -->

<!-- ## Hot Potato 🥔🥔, Threads, & Queues

### Summary of Issues & Solutions

Both sending to the database and doing inference on the same thread slowed down the entire process since internet outages affected sending to the database without waiting for ~10 seconds sometimes.

By having database operations on separate threads they would not block the main loop. To send them data, thread-safe queues were used. Instead of sending just one at a time batches were sent to decrease the amount of round trips to the database.

---

At the start of the project internet issues had been a concern top of mind because the free wifi was not very good. Low signal and inconsistent service plagued the system for months. Before landing on the current network solution implemented, I had added code changes to compensate for a lack of consistent network availability.

Originally when the system timed out sending information to the database everything would grind to a halt because sending and inference was done on the same thread. Simply put, this is like 3 people playing a game of hot potato but if Fred takes 5x as long to throw the potato the others have to wait; that is if he ever decides to throw the potato. Naturally this can cause issues when you play with more than one hot potato and all the angry fingers will be pointing at Fred.

To address Fred taking longer and the two others having to wait for him, imagine giving each hot potato thrower a basket and letting them continue throwing while Fred takes his time. We don't need to confine this idea to our imagination. In practice thread safe queues were used on different threads. With this change instead of waiting to send to the database, the system queues what it was going to send and a different thread later sends it to the database.

Back to the analogy, we are doing better because while Fred takes his time throwing that potato, the other players (with multiple potatoes) can keep throwing. As they keep throwing Fred can take from his basket and throw those where the basket is the queue and each player playing independently is being on different threads. Still, an issue remains… Fred can only throw one potato at a time so his basket keeps getting fuller. Let Fred throw 4 potatoes at a time and he spends less time turning around and reaching for potatoes. Technically this meant batching requests to the database to reduce the round trips.

After several attempts to fix the wifi signal received by the computer, we landed on our current solution but with caveats. -->

<!-- TODO:
* Dynamically choosing classes. 
* The binary class, where it went wrong. 
* rounding error
* could make faster by doing same wav files at once, if not already? -->


<!-- # Training The AI Model

... -->

# Credits

<div class="two-col">
<div>

#### Paulson Hanel
* Machine Learning (*the AI*). 
* Data labelling. 
* Local infastructure. 

My personal website is <a href="https://fizzyvermin.com/" target="_blank">here</a>, and my Instagram <a href="https://www.instagram.com/fizzyvermin/" target="_blank">here</a>. Currently there are no open sourced projects on my GitHub. 

<!-- https://github.com/Flogging0140 -->

</div>
<div>

#### Paxton Neustaeter
* App.
* Database. 
* Server Infastructure.

<!-- [GitHub](https://github.com/Odalith) -->
<a href="https://github.com/Odalith" target="_blank">GitHub</a>


</div>
</div>

**NOTE**: Some parts of the project did share overlap. "*It was great to work with Paxton, he contributed massively to this large endeavor, very spleandid*" - Paulson W. Hanel. 

---

**NOTE**: The project memes are <u>NOT</u> representative of how I feel or act in the work place. They are simply likely relatable memes to how others have felt at specific times. I have only 1 person to share these with, maybe slightly more now.