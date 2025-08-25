---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* M.S. in Computer Science (Emphasis in Machine Learning), University of Minnesota, 2025
* B.S. in Computer Science (Minors in Math and Neuroscience), University of Minnesota, 2023

Work experience
======
* July 2024 - Present: Software Engineer II, Tokyo Electron Limited
  * Architect, develop scheduling system (C#) for $5MM semiconductor manufacturing tool (C#, 25-50k LOC)
  * Improve throughput (8%), reduce fatal error rate (3%), save customers $1.3MM annually
  * Aid in development of XGBoost model (Matlab) to improve etch precision (14%) and reduce etch time (7%)
  * Built rotational-invariant CNN (PyTorch) to improve photolithography speed (save ~1.5 sec per wafer) 


* May 2024 - Aug 2024: Neuromodulation Software Engineering Intern II, Boston Scientific
  * Created streamlined framework for Agile development to enhance timeline, improve consistency
  * Slashed PM approval backlog by 42%, rejected PRs by 30%, raised sprint velocity by 15%
  * Built Jira Plugin (Java, Spring Boot), wrote scripts for ScriptRunner (Java, Groovy) to enforce framework
  * Collaborated with SWE Director, PMs, Devs, identified issues with existing Agile processes

* May 2023 - Aug 2023: Neuromodulation Software Engineering Intern, Boston Scientific
  * Developed iOS app (Swift) to control implant (C++), reduced patient visits (25%), visit duration (~45 min)
  * Utilized Azure Communications backend allowing patients and clinicians to chat and video/voice call
  * Collaborated in an Agile, cross-functional team to coordinate, implement project


  
Skills
======
* Python
  * PyTorch
  * TensorFlow
  * Jupyter
  * Pandas
  * Keras
  * SciKit
  * MNE (for EEG analysis/Signal Processing)
* C#
* JavaScript
  * React
  * Angular
  * Node.JS
* SQL
* C/C++
* R
* MATLAB

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
    
Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Service and leadership
======
* Currently signed in to 43 different slack teams
