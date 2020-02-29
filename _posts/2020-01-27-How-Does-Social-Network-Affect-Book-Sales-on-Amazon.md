---
layout: post
title: "How Does Social Network Affect Book Sales on Amazon?"
date: 2020-01-27
excerpt: "Using Social Network Analysis and Statistics to Understand Demand Pattern on Amazon Books"
tags: [Machine Learning, Predictive Modelling, Poisson Regression, R, igraph, Social Network Analysis]
comments: true
---


## Understanding The Effect of Social Network on Sales and Demand Spillover on Amazon Books
<img width="542" alt="Screen Shot 2020-02-28 at 7 00 38 PM" src="https://user-images.githubusercontent.com/54050356/75599764-b77af080-5a5c-11ea-9321-260ea0576c8a.png">

In this post I am going to uncover the effect that social network has on Amazon Book Sales. I have been particularly interested in <b>Social Network Analysis</b> given how broad and applicable the concepts are in today's businesses, especially in the tech or e-commerce industry. Recently, I had the opportunity to work on a project from one of my Graduate classes, where I got to analyze the networks or connections between each book sold on Amazon, identify the links or relationships in the network, and finally conclude how these links can affect the sales of each book. Alright, without further ado, let's take a look at the analysis!

## Data and Libraries

First, below are the libraries that I used for this analysis. The analysis was done in R and the required library to run social network analysis in R was igraph. 

    library(igraph)
    library(dplyr)
    library(sqldf)
    library(ggplot2)
    library(psych)

Now, Let's take a look into our dataset. There were 2 datasets that I used for this analysis. One was on product, which includes individual books related information such as their id, sales rank, and book title. The second dataset was about co-purchase, which contain Source and Target nodes for every book id. 

The salesrank in the product dataset has inverse relationship with book sales volume, in that the higher the salesrank, the lower the sales volume will be. Conversly, the lower the salesrank, the higher the sales volume will be. A study has proved that the relationship between Amazon book salesrank and sales volume is significant and that it is appropriate to use salesrank as a subsite of sales volume for the analysis. To access the full document on the study mentioned above, please visit this [link](https://kk.org/mt-files/reCCearch-mt/The_Effect_of_Word.pdf). 

The copurchase data tells us for every book id, what was the source book that pointed to that and in reverse, for every book id, which other book does it link to. Both dataset were from Amazon.

To limit the data and for the sake of the analysis, I have filtered the data to only include books with Sales Rank less than 150K. 

#### Product Data

    head(product1)

    ##    id
    ## 12 12
    ## 33 33
    ## 39 39
    ## 45 45
    ## 74 74
    ## 77 77
    ##                                                                                                       title
    ## 12 Fantastic Food with Splenda : 160 Great Recipes for Meals Low in Sugar, Carbohydrates, Fat, and Calories
    ## 33                                                                           Double Jeopardy (T*Witches, 6)
    ## 39                                                                           Night of Many Dreams : A Novel
    ## 45                                                                     Beginning ASP.NET Databases using C#
    ## 74                                                      Service Delivery (It Infrastructure Library Series)
    ## 77                                                                                     Water Touching Stone
    ##    group salesrank review_cnt downloads rating
    ## 12  Book     24741         12        12    4.5
    ## 33  Book     97166          4         4    5.0
    ## 39  Book     57186         22        22    3.5
    ## 45  Book     48408          4         4    4.0
    ## 74  Book     27507          2         2    4.0
    ## 77  Book     27012         11        11    4.5

#### Copurchase Data

    head(purch1)
    
    ##     Source Target
    ## 50      12    261
    ## 357     74    282
    ## 369     77    422
    ## 381     79     82
    ## 565    117    131
    ## 582    120    439

# Plotting The Social Networks

Now that we've understand how the data look like, let's move on to the fun part.. Visualization! 
Below, we will see the network and connections that were formed between these books. Using the igraph library, we can turn our dataframe into graph data frame that will treat all the book ids as a <b>Node</b> and based on the source and target column, we can identify the <b>in and out degree</b>. Each book id in the source column will have an out degree to the book in the target column. Likewise, every book in target column will have an in degree from a book in the source column.

With the steps below, I identified book id or a node that has the highest degree. In other words, this book has the most total links (in + out) in the network. Based on the result, book 33 and 4429 had the same total in and out degree. However, when I removed the salesrank filter or in the full dataset, book 33 has a lot more sources pointing to it. Therefore, I picked book id = 33 with 53 degrees total. Book 33 happened to be a thriller book called, Double Jeopardy while Book 4429 was a Harley-Davidson Panheads book. Before we move further, keep in mind that this book has a quite high salesrank, which indicated low sales volume. (We will get more into the demand topic later on in this analysis)

    net1 <- graph.data.frame(purch1, directed=T)

    ## 2. Create a variable named in-degree
    in_degree <- degree(net1, mode="in")
    head(in_degree)

    ##  12  74  77  79 117 120 
    ##   5   1   3   0   9   3

    # 3. Create a variable named out-degree
    out_degree <- degree(net1, mode="out")
    all_degree <- degree(net1, mode="all")
    all_degree[all_degree == max(all_degree)]

    ## 4429   33 
    ##   53   53

    ## 4. Choosing book id = 33
         id                                    title group salesrank review_cnt downloads rating
         33           Double Jeopardy (T*Witches, 6)  Book     97166          4         4    5.0
       4429 Harley-Davidson Panheads, 1948-1965/M418  Book    147799          3         3    4.5

#### Book 33
<img width="1001" alt="Screen Shot 2020-02-28 at 7 12 44 PM" src="https://user-images.githubusercontent.com/54050356/75599931-62d87500-5a5e-11ea-9556-7527e2c4e156.png">

#### Book 4429
<img width="1080" alt="Screen Shot 2020-02-28 at 7 28 38 PM" src="https://user-images.githubusercontent.com/54050356/75600126-8f8d8c00-5a60-11ea-9bb4-179ea7f5932c.png">

Now, let's get all the subcomponents of book = 33 (in other words, all books that were connected to book 33 in the network) and plot the network.

    sub <- subcomponent(net1, "33", mode = "all")

    # 5. Visualize the subcomponent
    graph <- induced_subgraph(net1, sub)
    V(graph)$label <- V(graph)$name
    V(graph)$degree <- degree(graph)

    set.seed(222)
    plot(graph,
         vertex.color=rainbow(33),
         vertex.size=V(graph)$degree*0.08,
         edge.arrow.size=0.01,
         vertex.label.cex=0.03,
         layout=layout.fruchterman.reingold)

<img width="453" alt="Screen Shot 2020-02-28 at 7 20 15 PM" src="https://user-images.githubusercontent.com/54050356/75600040-9667cf00-5a5f-11ea-8fcc-d91ecfae9515.png">

Overall, this was how the network looked like. The big 2 nodes in the center indicate nodes with highest total in and out degree. The big nodes on the right was the Double Jeopary Book (33) and the other one on the left was the Harley Davidson Book (4429). We can see that both of these big nodes have their own community of books formed around them. The nides closer to Book 33 and 4429 are closer to each other than the nodes that are further away from the center. This means that the influence power within these nodes closer to the center are relatively high. 

Both of these books were connected by a local bridge link which indicated a weak connection and low influence power yet served as the link where most information transfers take place. The information flow from all the nodes on one side of the bridge have to go through the bridge to get to the other side. Therefore, given that these two main nodes are connected, there can be demand spillover from people who purchased the books around or at node 33 to the books around or at node 4429. 
The bridge link is highly related to the concept of the "strength of weak ties" where more novel information were passed on through weak ties rather than strong ties.

Let's visualize the clusters in this network. Looking at the plot below, we can see that there were multiple subclusters formed inside of the main clusters due to the degree or link density of each these small nodes. 

<img width="542" alt="Screen Shot 2020-02-28 at 7 00 38 PM" src="https://user-images.githubusercontent.com/54050356/75599764-b77af080-5a5c-11ea-9321-260ea0576c8a.png">

We can see how spread out the network is using the diameter measurement. The <b>pink nodes</b> below indicated the <b>diameter nodes</b>. As shown below, The Harley Davidson Book was one of the diameter node (4429). The node was the furthest out in the diameter was node 37895, which was a romantic genre book (Book Title : Sons and Lovers (Signet Classics (Paperback)). As we can see from the book title, book 37895 was highly unrelated with Double Geopardy book thus it was on the edge of the network.

<img width="509" alt="Screen Shot 2020-02-28 at 8 01 25 PM" src="https://user-images.githubusercontent.com/54050356/75600549-5572b900-5a65-11ea-9845-794f8277fd67.png">

<img width="347" alt="Screen Shot 2020-02-28 at 8 02 41 PM" src="https://user-images.githubusercontent.com/54050356/75600553-66232f00-5a65-11ea-8a4e-31ebb123bffd.png">

    #Diameter Nodes
    print(as_ids(diam))

    ##  [1] "37895" "27936" "21584" "10889" "11080" "14111" "4429"  "2501"  "3588" 
    ## [10] "6676"

    diameterbooks <- product[product$id %in%  as_ids(diam),]
       id                                                      title group salesrank review_cnt downloads rating
     2501          The Narcissistic Family : Diagnosis and Treatment  Book      9727         19        19    5.0
     3588                     A Fourth Treasury of Knitting Patterns  Book     91126          1         1    5.0
     4429                   Harley-Davidson Panheads, 1948-1965/M418  Book    147799          3         3    4.5
     6676                                             Song of Eagles  Book    130216          1         1    5.0
    10889                                 Sixpence Bride (Timeswept)  Book     96977         16        16    4.5
    11080 Counter Intelligence: Where to Eat in the Real Los Angeles  Book     28673         13        13    5.0
    14111                    Memories, Dreams, Reflections (Vintage)  Book      4818         38        38    4.5
    21584                                           A Year and a Day  Book    107460         52        52    4.0
    27936                     Numerology For Personal Transformation  Book    111939          1         1    5.0
    37895              Sons and Lovers (Signet Classics (Paperback))  Book      9236         70        70    4.0
 
To fully analyze and quantify the effect of these links between each nodes in the network, there are numerous statistics that we can compute. For this analysis, I focused on the diameter, edge density and distance. I also analyzed the network centrality measures with methodology such as degree centrality, closeness, betweeness, eigen centrality, hub and authority scores.

    diameter <- diameter(graph, directed=T, weights=NA)
    edge_density <- edge_density(graph, loops=F)
    mean_distance <- mean_distance(graph, directed=T)
    data.frame(Statistics = c("diameter","edge_density","mean_distance"),
                  Value = c(diameter,edge_density,mean_distance))

    ##      Statistics       Value
    ## 1      diameter 9.000000000
    ## 2  edge_density 0.001436951
    ## 3 mean_distance 2.167236662

Degree Centrality

    degree_centrality <- centr_degree(graph, mode="all")
    paste("degree centrality - centralization:",degree_centrality$centralization)

    ## [1] "degree centrality - centralization: 0.027940579512858"

Closeness

    closeness <- closeness(graph, mode="all", weights=NA) 
    head(sort(closeness, decreasing = TRUE))

    ##           33          626       242813         4429          224         2558 
    ## 0.0001612383 0.0001585289 0.0001571092 0.0001557632 0.0001496110 0.0001478852

Betweeness

    between <- betweenness(graph, directed=T, weights=NA)
    head(sort(between, decreasing = TRUE))

    ##  2501  4429  3588 31513 30106 60266 
    ##   298   260   150    92    64    62
    
Eigen Centrality    
    
    eigen_centrality <- eigen_centrality(graph, directed=T, weights=NA)
    head(sort(eigen_centrality$vector, decreasing = TRUE))
    ## 8160  26268  39157  26267    302  46301 
    ## 1.0000 0.6340 0.6340 0.6340 0.5000 0.4641 

Hub Score

    hub_score <- hub.score(graph)$vector
    head(sort(hub_score), descreasing=TRUE)

    ## 1817 2071 2505 3032 3119 3588 
    ##    0    0    0    0    0    0

Authority Score

    authority_score <- authority.score(graph)$vector
    head(sort(authority_score), descreasing=TRUE)

    ##  626 2423 2501 4429 7325 7544 
    ##    0    0    0    0    0    0
    

Degree Distribution and Cummulative Frequency Distribution
<img width="543" alt="Screen Shot 2020-02-28 at 8 39 23 PM" src="https://user-images.githubusercontent.com/54050356/75600924-7db0e680-5a6a-11ea-82dd-4fe1fd1c128f.png">
<img width="545" alt="Screen Shot 2020-02-28 at 8 39 46 PM" src="https://user-images.githubusercontent.com/54050356/75600926-899ca880-5a6a-11ea-96ed-8feeb1f45760.png">

The degree distribution followed a “power law distribution”, or in other words, it showed a skewed distribution with a long right tail where there were a lot of nodes that have only a few links. Since Book 33 and 4429 had a lot of degree, they are at the tail of this distribution. Therefore, if we randomly remove a node from the network, the probability of removing the nodes with a high number of degrees are very low. In this case, we definitely do not want to remove book 4429 and 33 since they hold the network together. 

Additionally, the low edge density does not necessarily indicate that the information transferability is slow amongst these nodes, or in this context, the low edge density does not indicate slow demand spillover between these books. This is due to the fact that the bigger the group, the more likely the density to be low.

Moreover, the mean distance was relatively small. This indicated that the nodes were on average closely positioned to each other. In a social network with relatively small mean distance, the information transfer and influence power are relatively strong.

Looking at the centrality measures, we can tell which node was more important in maintaining the network than others. In our case, we can see in the closeness measure that the node 33 has the highest closeness score. Meaning that information transfer was faster from this node to other nodes. Therefore, there will be a demand spillover to other nodes around node 33 once someone purchases book 33. Besides closeness, I also measured the eigen centrality. Just like degree centrality, eigen centrality measures how many other nodes a particular node is connected to. However, it takes the calculation a step further by also consider the connection of the connected nodes and using that information to assign weight to those connected nodes.  A more central neighborhood node gets assigned higher weight.

Based on the betweenness measure, we can see that node 2501 and 4429 were the top 2 nodes that a lot of these subcomponent nodes needed to pass through to get to the other nodes. This indicated that node 2501 and 4429 are important in the network to connect other nodes to each other. 

# Merging Analysis the Output Together

Since the goal of the analysis is to estimate how these social network aspects affect the demand on these books, we would need to compute a predictive model for that analysis. Therefore, I began by merging the result of the computed statistics on the social network analysis to my product1 dataframe.

    # 7. Create neighbors variable
    names(purch1)[1] <- "id"
    sid <- as_ids(sub)
    sub_prod <- product1[product1$id %in% sid,]
    neighbors_mean_variables <- as.data.frame(purch1 %>%
                                  group_by(Target) %>%
                                  inner_join(sub_prod, by="id") %>%
                                  summarize(nghb_mn_rating = mean(rating),
                                        nghb_mn_salesrank = mean(salesrank),
                                        nghb_mn_review_cnt=mean(review_cnt)))
                      


    in_df <- data.frame(id = names(in_degree), in_degree)
    out_df <- data.frame(id = names(out_degree), out_degree)
    closeness <- data.frame(id = names(closeness), closeness)
    betweenness <- data.frame(id = names(between), between)
    authority <- data.frame(id = names(authority_score), authority_score)
    hub <- data.frame(id = names(hub_score), hub_score)

    in_df$id<-as.numeric(as.character(in_df$id))
    out_df$id<-as.numeric(as.character(out_df$id))
    closeness$id<-as.numeric(as.character(closeness$id))
    betweenness$id<-as.numeric(as.character(betweenness$id))
    authority$id<-as.numeric(as.character(authority$id))
    hub$id<-as.numeric(as.character(hub$id))

    names(neighbors_mean_variables)[1] <-"id"

    data <- sub_prod %>% inner_join(neighbors_mean_variables, by = "id") 
    data <- data  %>% inner_join(in_df, by = "id") 
    data <- data  %>% inner_join(out_df, by = "id") 
    data <- data %>% inner_join(closeness, by="id") 
    data <- data %>% inner_join(betweenness, by="id") 
    data <- data %>% inner_join(authority, by="id") 
    data <- data %>% inner_join(hub, by="id")

The final data for the model looked like below. It had 9 additional columns with information such as: 
- neighborhood mean rating (nghb_mn_rating) or the average rating for the nodes within the neighborhood of each book.
- neighborhood mean salesrank (nghb_mn_salesrank) or the average salesrank for the nodes within the neighborhood of each book.
- neighborhood mean review count (nghb_mn_review_cnt) or the average review count for the nodes within the neighborhood of each book.
- Additionally, it has the social network statistics above.

        ##    id
        ## 1  33
        ## 2  77
        ## 3  78
        ## 4 130
        ## 5 148
        ## 6 187
        ##                                                                                    title
        ## 1                                                         Double Jeopardy (T*Witches, 6)
        ## 2                                                                   Water Touching Stone
        ## 3                                                 The Ebony Cookbook: A Date With a Dish
        ## 4 The O'Reilly Factor: The Good, the Bad, and the Completely Ridiculous in American Life
        ## 5                                                                               Firebird
        ## 6                         Words for Smart Test Takers (Academic Test Preparation Series)
        ##   group salesrank review_cnt downloads rating nghb_mn_rating nghb_mn_salesrank
        ## 1  Book     97166          4         4    5.0       4.103774          82153.26
        ## 2  Book     27012         11        11    4.5       4.666667          41744.00
        ## 3  Book    140480          3         3    4.5       4.500000          73179.00
        ## 4  Book     29460        375       375    3.5       4.500000          19415.00
        ## 5  Book     77008         42        42    4.0       0.000000          46701.00
        ## 6  Book     17104          4         4    5.0       4.500000         133546.67
        ##   nghb_mn_review_cnt in_degree out_degree    closeness between authority_score
        ## 1          21.075472        53          0 1.612383e-04       0    1.000000e+00
        ## 2           4.000000         3          1 9.045681e-05      12    4.449831e-17
        ## 3         157.818182        11          0 1.191753e-04       0    5.753636e-04
        ## 4           6.000000         1          1 1.077935e-04       1    2.473186e-17
        ## 5           0.000000         1          1 1.009897e-04       2    2.567663e-17
        ## 6           3.666667         3          3 1.076774e-04       2    2.431071e-05
        ##      hub_score
        ## 1 0.000000e+00
        ## 2 2.239872e-16
        ## 3 1.140518e-17
        ## 4 5.531568e-04
        ## 5 3.592652e-05
        ## 6 5.989914e-04

# Running Poisson Regression to Predict The Book Salesrank

Before running the model, let's look at each of the variables in the dataset. Based on the summary statistics, we can definitely see that there were a lot of variables that were highly skewed. We can see it visually as well that variable <b>closeness and nghb_mn_salesrank</b>  were the only variables that looked normally distributed.

    ##                    vars   n     mean       sd   median  trimmed      mad  min
    ## salesrank             1 518 70850.97 45410.37 68466.50 69754.01 59099.40   64
    ## review_cnt            2 518    26.34    72.68     6.00    10.65     7.41    0
    ## downloads             3 518    26.26    72.65     6.00    10.56     7.41    0
    ## rating                4 518     3.94     1.46     4.50     4.29     0.74    0
    ## nghb_mn_rating        5 518     3.88     1.32     4.33     4.15     0.62    0
    ## nghb_mn_salesrank     6 518 73422.41 37494.73 73840.15 72895.76 43153.08 1596
    ## nghb_mn_review_cnt    7 518    25.58    68.32     8.00    11.83     8.90    0
    ## in_degree             8 518     2.26     3.89     1.00     1.56     0.00    1
    ## out_degree            9 518     1.35     0.85     1.00     1.33     1.48    0
    ## closeness            10 518     0.00     0.00     0.00     0.00     0.00    0
    ## between              11 518     6.59    20.53     2.00     3.13     2.97    0
    ## authority_score      12 518     0.00     0.04     0.00     0.00     0.00    0
    ## hub_score            13 518     0.04     0.19     0.00     0.00     0.00    0
    ##                       max  range  skew kurtosis      se
    ## salesrank          149844 149780  0.16    -1.28 1995.22
    ## review_cnt           1015   1015  7.10    72.99    3.19
    ## downloads            1015   1015  7.11    73.12    3.19
    ## rating                  5      5 -1.91     2.55    0.06
    ## nghb_mn_rating          5      5 -1.86     2.79    0.06
    ## nghb_mn_salesrank  149844 148248  0.10    -0.83 1647.42
    ## nghb_mn_review_cnt   1015   1015  8.40    98.06    3.00
    ## in_degree              53     52  9.20   107.78    0.17
    ## out_degree              4      4  0.30     0.07    0.04
    ## closeness               0      0  0.04    -0.56    0.00
    ## between               298    298 10.16   125.83    0.90
    ## authority_score         1      1 22.42   504.54    0.00
    ## hub_score               1      1  4.77    20.85    0.01

<img width="1011" alt="Screen Shot 2020-02-28 at 9 12 30 PM" src="https://user-images.githubusercontent.com/54050356/75601350-12b5de80-5a6f-11ea-84a8-cb10558eba23.png">

#### Running The Models

As mentioned above, I used poisson regression to estimate the salesrank given that salesrank is a count data and there were multiple books having the same salesrank. The dependent variable will be salesrank, and all other variables will be independent.

#### Model 1
In the first model, I included all the variables as is just to see how it performed. The result indicated that almost all variables except for closeness were significant to determine the salesrank of the book. However, the AIC was pretty high and had rooms for improvements.

    Deviance Residuals: 
     Min      1Q  Median      3Q     Max  
    -370.8  -160.0    -8.7   122.1   522.0  

    Coefficients:
                               Estimate      Std. Error  z value             Pr(>|z|)    
    (Intercept)              11.17702215504   0.00110466751 10118.00 < 0.0000000000000002 ***
    review_cnt               -0.02796466091   0.00018801201  -148.74 < 0.0000000000000002 ***
    downloads                 0.02380425972   0.00018815296   126.52 < 0.0000000000000002 ***
    rating                   -0.00634948813   0.00010974014   -57.86 < 0.0000000000000002 ***
    in_degree                 0.01138846743   0.00007092589   160.57 < 0.0000000000000002 ***
    out_degree                0.08326014263   0.00021449805   388.16 < 0.0000000000000002 ***
    closeness               -10.59983144669   7.84379908137    -1.35                 0.18    
    between                  -0.00187610074   0.00001186298  -158.15 < 0.0000000000000002 ***
    hub_score                 0.22333468517   0.00086024101   259.62 < 0.0000000000000002 ***
    authority_score          -0.22656835179   0.00484944821   -46.72 < 0.0000000000000002 ***
    nghb_mn_review_cnt        0.00070560244   0.00000198202   356.00 < 0.0000000000000002 ***
    nghb_mn_salesrank         0.00000003477   0.00000000451     7.72    0.000000000000012 ***
    nghb_mn_rating           -0.01283122704   0.00012470470  -102.89 < 0.0000000000000002 ***
    eigen_centrality.vector  -1.52987902187   0.00355836448  -429.94 < 0.0000000000000002 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    (Dispersion parameter for poisson family taken to be 1)

    Null deviance: 16968896  on 517  degrees of freedom
    Residual deviance: 15060277  on 504  degrees of freedom
    AIC: 15066856

    Number of Fisher Scoring iterations: 5

#### Model 2

With the findings from the first model, I created model 2 while removing the variable closeness and conducting a log transformations on all the other variables except for nghb_mn_salesrank due to the skewness of the data. The result showed a much improved AIC from the first model. Looking at the deviance residuals, the min and max were also more aligned as compared to the first model. Therefore, I chose this model as the final model. Now, let's interpret the result!

    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -337.3  -164.7   -18.9   120.4   394.9  

    Coefficients:
                                      Estimate Std. Error z value            Pr(>|z|)    
    (Intercept)                      10.891289   0.002819  3863.3 <0.0000000000000002 ***
    log(review_cnt + 1)              -0.444245   0.003262  -136.2 <0.0000000000000002 ***
    log(downloads + 1)                0.274481   0.003265    84.1 <0.0000000000000002 ***
    log(rating + 1)                   0.152331   0.000344   442.2 <0.0000000000000002 ***
    log(in_degree + 1)               -0.060936   0.000465  -131.0 <0.0000000000000002 ***
    log(out_degree + 1)               0.159693   0.000546   292.6 <0.0000000000000002 ***
    log(between + 1)                 -0.015351   0.000195   -78.7 <0.0000000000000002 ***
    log(hub_score + 1)                0.238475   0.001121   212.7 <0.0000000000000002 ***
    log(authority_score + 1)          0.606572   0.005093   119.1 <0.0000000000000002 ***
    log(nghb_mn_review_cnt + 1)       0.059945   0.000141   425.3 <0.0000000000000002 ***
    log(nghb_mn_salesrank + 1)        0.027622   0.000237   116.7 <0.0000000000000002 ***
    nghb_mn_rating                   -0.027886   0.000132  -210.8 <0.0000000000000002 ***
    log(eigen_centrality.vector + 1) -1.722708   0.004586  -375.7 <0.0000000000000002 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    (Dispersion parameter for poisson family taken to be 1)

        Null deviance: 16968896  on 517  degrees of freedom
    Residual deviance: 14773532  on 505  degrees of freedom
    AIC: 14780109

    Number of Fisher Scoring iterations: 5

#### Interpretation

Based on Model 2 result, we can see that the increase in number of reviews, number of in degree, the betweeness, average neighborhood rating can bring the salesrank down thus will bring the sales volume up. The interesting note here was that the eigen centrality significantly reduced the estimated salesrank. This indicated that the more "popular" books that a book was linked-to, the higher the estimated increase in the demand for that book. In this case "popular" means that the connected book was also linked to a lot of other books. Below was the estimate of the increase in the salesrank for every 1% increase in each variable when all other variables remain constant.

     > print(((1.01**coef(p2)[2:13])-1)*100)
                 log(review_cnt + 1)               log(downloads + 1) 
                            -0.44106                          0.27349 
                     log(rating + 1)               log(in_degree + 1) 
                             0.15169                         -0.06062 
                 log(out_degree + 1)                 log(between + 1) 
                             0.15903                         -0.01527 
                  log(hub_score + 1)         log(authority_score + 1) 
                             0.23757                          0.60538 
         log(nghb_mn_review_cnt + 1)       log(nghb_mn_salesrank + 1) 
                             0.05966                          0.02749 
         log(eigen_centrality.vector + 1) 
                            -1.69954 

Interestingly, an increase in rating was associated with a 15% increase in the salesrank. This indicated that having higher ranking did not mean that the book would have more sales. Rather, the number of reviews was found to be more relevant to increase the sales (or reduce the salesrank). Based on the above, every 1% increase in the number of reviews, the salesrank was estimated to decrease by 44%. Although having a small effect, in degree did play a role in the estimated decrease of salesrank. This indicated that the more the book is linked-to by other books, there would be a demand spillover.

# Conclusion

To conclude, social network is a powerful fuel for today's business. Without social network, it is almost impossible for a business to grow. Given how import it is, understanding the effect and how to manage these networks become even more crucial. This post is just a POC of what we can do with social network analysis. In the future, I hope to use more of these analysis to analyze clusters and social influences in customer behavior. I am also interested in applying this concept to social science projects. I hope to expand and utilize this methodologies more in the near future. 

Thank you for reading through this post! As usual, I appreciate any feedbacks/suggestions/questions.

# References
 https://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch02.pdf
 https://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch03.pdf
 https://cs.hse.ru/data/2015/04/30/1098187988/1._tutorial_igraph.pdf

