# Writer Identifcation System
A system that is used to identify a writer from a handwritten script. In this system, We used a texture based feature extractor called local binary pattern and SVM classifier to distinguish between different writers. We trained and tested our project using IAM dataset. 
# How it works
1. Install project dependencies from requirements.txt
2. Add the test cases in folder `/data/TestCases`
3. Add the expected results in file `/inputs/actualt_results.txt`
4. Open terminal in the `'/src'` directory then Run using this command
```Console
python main.py
```
5. Finally, the results will be in `/outputs` diectory in two files time.txt and results.txt
