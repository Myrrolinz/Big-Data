# NKU Big Data Course

This repository is the code for NKU Big Data Course. The assignments consist of two main parts.

### Part 1: Page Rank

Dataset: Data.txt

The format of the lines in the file is as follow:

**FromNodeID ToNodelD**

In this project, you need to report the Top 100 NodelD with their PageRank scores. You can choose different parameters, such as the teleport parameter, to compare different results.

One result you must report is that when setting the teleport parameter to 0.85.

In addition to the basic **PageRank algorithm**, you need to implement the **Block-Stripe Update algorithm**.

### Part 2: Recommendation System

Task: 

Predict the rating scores of the pairs (u, i) in the Test.txt file. 

Dataset (data-202205.zip)： 

(1)	Train.txt, which is used for training your models. 

(2)	Test.txt, which is used for test. 

(3)	ItemAttribute.txt, which is used for training your models (optional). 

(4) ResultForm.txt, which is the form of your result file. 

The formats of datasets are explained in the DataFormatExplanation.txt. 

Note that if you can use ItemAttribute.txt appropriately and improve the performance of the algorithms, **additional points (up to 10)** can be added to your final course score. 

In this project, you need to report the predicted rating scores of the unknown pairs (u, i) in the Test.txt file. You can use any algorithms you have learned from the course or from other resources (such as MOOC). 

One group (consisting of at most three students) needs to write a report about this project. The report should include but not limited to the following contents: 

1. Basic statistics of the dataset (e.g., number of users, number of ratings, number of items, etc); 
2. Details of the algorithms; 
3. Experimental results of the recommendation algorithms (RMSE, training time, space consumption); 
4. Theoretical analysis or/and experimental analysis of the algorithms. 