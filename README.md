# MTVCrafter: 4D Motion Tokenization for General Human Image Animation

> Official project page of **MTVCrafter**, a novel framework for general and high-quality human image animation using raw 3D motion.

## 🔍 Abstract

Human image animation has attracted increasing attention and developed rapidly due to its broad applications in digital humans. However, existing methods rely on 2D rendered pose images for motion guidance, which limits generalization and discards essential 3D information.  
To tackle these problems, we propose **MTVCrafter (Motion Tokenization Video Crafter)**, the first framework that directly models raw 3D motion sequences for human image animation beyond intermediate 2D representations.

- We introduce **4DMoT (4D motion tokenizer)** to encode raw motion data into discrete motion tokens, preserving compact but expressive spatio-temporal information.
- Then, we propose **MV-DiT (Motion-aware Video DiT)**, which integrates a motion attention module and 4D positional encodings to effectively modulate vision tokens with motion tokens.
- The overall pipeline facilitates high-quality human video generation guided by 4D motion tokens.

MTVCrafter achieves **state-of-the-art results with an FID-VID of 6.98**, outperforming the second-best by approximately **65%**. It generalizes well to diverse characters (single/multiple, full/half-body) across various styles.

## 🎯 Motivation

<img src="./static/images/motivation.png" alt="Motivation" width="100%">

## 🎞️ Animation Results

> All videos below autoplay and loop automatically. You may hover or click to replay if your browser blocks autoplay.

### Luffy
<video src="./static/videos/luffy.mp4" autoplay loop muted playsinline width="320"></video>

### Mona Lisa
<video src="./static/videos/monalisa.mp4" autoplay loop muted playsinline width="320"></video>

### Siren
<video src="./static/videos/siren.mp4" autoplay loop muted playsinline width="320"></video>

### Tanjiro
<video src="./static/videos/tanjianci.mp4" autoplay loop muted playsinline width="320"></video>

### Pixel Style
<video src="./static/videos/xiangsu_new.mp4" autoplay loop muted playsinline width="320"></video>

### Daji
<video src="./static/videos/daji.mp4" autoplay loop muted playsinline width="320"></video>

### Spiderman
<video src="./static/videos/spider.mp4" autoplay loop muted playsinline width="320"></video>

### Ghibli Style
<video src="./static/videos/jibuli.mp4" autoplay loop muted playsinline width="320"></video>

### Cowboy
<video src="./static/videos/niuzai.mp4" autoplay loop muted playsinline width="320"></video>

### Human
<video src="./static/videos/human.mp4" autoplay loop muted playsinline width="320"></video>

### Ultraman
<video src="./static/videos/dijia.mp4" autoplay loop muted playsinline width="320"></video>

### Iron Man
<video src="./static/videos/iron-man.mp4" autoplay loop muted playsinline width="320"></video>

---

## 📄 Citation

Coming soon.

## 📬 Contact

For questions or collaboration, feel free to reach out via GitHub Issues or Email.

---

<p align="center">
  <img src="./static/images/favicon.svg" alt="MTVCrafter Logo" width="40">
</p>
