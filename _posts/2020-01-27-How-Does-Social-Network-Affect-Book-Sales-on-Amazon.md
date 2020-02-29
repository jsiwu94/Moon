---
layout: post
title: "How Does Social Network Affect Book Sales on Amazon?"
date: 2020-01-27
excerpt: "Using Social Network Analysis and Statistics to Understand Demand Pattern on Amazon Books"
tags: [Machine Learning, Predictive Modelling, Poisson Regression, R, igraph, Social Network Analysis]
comments: true
---


## Understanding The Effect of Social Network on Sales and Demand Spillover for Amazon Books
<img width="542" alt="Screen Shot 2020-02-28 at 7 00 38 PM" src="https://user-images.githubusercontent.com/54050356/75599764-b77af080-5a5c-11ea-9321-260ea0576c8a.png">

In this post I am going to uncover the effect that social network has on Amazon Book Sales. I have been particularly interested in of <b>Social Network Analysis</b> given how broad and applicable the concepts are in today's businesses, especially in the tech or e-commerce industry. Recently,I had the opportunity to work on a project from one of my Graduate classes, where I got to analyze the network or connection between each book sold on Amazon, identify the links or relationships in the network, and finally conclude how these links can affect the sales of each book. Alright, without further ado, let's take a look at the analysis!

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

We can see how spread out the network is using the diameter measurement. 

<img width="509" alt="Screen Shot 2020-02-28 at 8 01 25 PM" src="https://user-images.githubusercontent.com/54050356/75600549-5572b900-5a65-11ea-9845-794f8277fd67.png">

<img width="347" alt="Screen Shot 2020-02-28 at 8 02 41 PM" src="https://user-images.githubusercontent.com/54050356/75600553-66232f00-5a65-11ea-8a4e-31ebb123bffd.png">


    #analyze the Diameter Nodes
    print(as_ids(diam))

    ##  [1] "37895" "27936" "21584" "10889" "11080" "14111" "4429"  "2501"  "3588" 
    ## [10] "6676"

    diameterbooks <- product[product$id %in%  as_ids(diam),]
    diameterbooks[order(-diameterbooks$salesrank), ]

    ##          id                                                      title group
    ## 4390   4429                   Harley-Davidson Panheads, 1948-1965/M418  Book
    ## 6608   6676                                             Song of Eagles  Book
    ## 27613 27936                     Numerology For Personal Transformation  Book
    ## 21376 21584                                           A Year and a Day  Book
    ## 10790 10889                                 Sixpence Bride (Timeswept)  Book
    ## 3558   3588                     A Fourth Treasury of Knitting Patterns  Book
    ## 10980 11080 Counter Intelligence: Where to Eat in the Real Los Angeles  Book
    ## 2481   2501          The Narcissistic Family : Diagnosis and Treatment  Book
    ## 37464 37895              Sons and Lovers (Signet Classics (Paperback))  Book
    ## 13976 14111                    Memories, Dreams, Reflections (Vintage)  Book
    ##       salesrank review_cnt downloads rating
    ## 4390     147799          3         3    4.5
    ## 6608     130216          1         1    5.0
    ## 27613    111939          1         1    5.0
    ## 21376    107460         52        52    4.0
    ## 10790     96977         16        16    4.5
    ## 3558      91126          1         1    5.0
    ## 10980     28673         13        13    5.0
    ## 2481       9727         19        19    5.0
    ## 37464      9236         70        70    4.0
    ## 13976      4818         38        38    4.5

Now let’s measure the statistics. Insight:

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

Looking at the Degree Distribution

    all_degree_df <- data.frame(id = names(all_degree), degree=all_degree)
    qplot(all_degree_df$degree,
          geom="histogram",
          binwidth = 0.5,  
          main = "Degree Distribution", 
          xlab = "Degree",  
          ylab="Frequency",
          fill=I("blue"), 
          col=I("red"), 
          alpha=I(.2),
          xlim=c(0,20))

    ## Warning: Removed 6 rows containing non-finite values (stat_bin).

    ## Warning: Removed 2 rows containing missing values (geom_bar).

![](Social_Network_Analysis_files/figure-markdown_strict/unnamed-chunk-12-1.png)

Cummulative Frequency Dist

    ggplot(all_degree_df, aes(degree, colour = "skyblue")) + stat_ecdf() +
            ggtitle("Cummulative Frequency Distribution") +
            ylab("Cummulative Frequency")

![](Social_Network_Analysis_files/figure-markdown_strict/unnamed-chunk-13-1.png)

Merging the data together and adding social network analysis into the
main data

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


    head(data)

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

looking at the summary stat for each variable

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

![](Social_Network_Analysis_files/figure-markdown_strict/unnamed-chunk-15-1.png)
Model 1

    p1 <- glm(salesrank ~ review_cnt+downloads+
                      rating+in_degree+
                      out_degree+closeness+between+
                      hub_score+authority_score+
                      nghb_mn_review_cnt+nghb_mn_salesrank+
                      nghb_mn_rating
              , data = data, family = "poisson")
    summary(p1)

    ## 
    ## Call:
    ## glm(formula = salesrank ~ review_cnt + downloads + rating + in_degree + 
    ##     out_degree + closeness + between + hub_score + authority_score + 
    ##     nghb_mn_review_cnt + nghb_mn_salesrank + nghb_mn_rating, 
    ##     family = "poisson", data = data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -363.25  -160.45    -7.61   122.01   519.58  
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error   z value Pr(>|z|)    
    ## (Intercept)         1.119e+01  1.108e-03 10096.697   <2e-16 ***
    ## review_cnt         -2.868e-02  1.877e-04  -152.749   <2e-16 ***
    ## downloads           2.457e-02  1.879e-04   130.759   <2e-16 ***
    ## rating             -7.061e-03  1.098e-04   -64.314   <2e-16 ***
    ## in_degree           2.801e-03  6.819e-05    41.069   <2e-16 ***
    ## out_degree          5.646e-02  2.057e-04   274.476   <2e-16 ***
    ## closeness          -1.789e+01  7.874e+00    -2.272   0.0231 *  
    ## between            -7.349e-04  1.111e-05   -66.157   <2e-16 ***
    ## hub_score           2.452e-01  8.593e-04   285.400   <2e-16 ***
    ## authority_score     1.895e-01  4.754e-03    39.861   <2e-16 ***
    ## nghb_mn_review_cnt  7.386e-04  1.969e-06   375.165   <2e-16 ***
    ## nghb_mn_salesrank   2.057e-07  4.498e-09    45.733   <2e-16 ***
    ## nghb_mn_rating     -9.723e-03  1.253e-04   -77.613   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 16968896  on 517  degrees of freedom
    ## Residual deviance: 15315200  on 505  degrees of freedom
    ## AIC: 15321778
    ## 
    ## Number of Fisher Scoring iterations: 5

Model 2

    p2 <- glm(salesrank ~ log(review_cnt+1)+log(downloads+1)+
                            log(rating+1)+log(in_degree+1)+
                            log(out_degree+1)+closeness+log(between+1)+
                            log(hub_score+1)+log(authority_score+1)+
                      log(nghb_mn_review_cnt+1)+log(nghb_mn_salesrank+1)+
                      log(nghb_mn_rating+1)
                      , data = data, family = "poisson")

    summary(p2)

    ## 
    ## Call:
    ## glm(formula = salesrank ~ log(review_cnt + 1) + log(downloads + 
    ##     1) + log(rating + 1) + log(in_degree + 1) + log(out_degree + 
    ##     1) + closeness + log(between + 1) + log(hub_score + 1) + 
    ##     log(authority_score + 1) + log(nghb_mn_review_cnt + 1) + 
    ##     log(nghb_mn_salesrank + 1) + log(nghb_mn_rating + 1), family = "poisson", 
    ##     data = data)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -334.64  -165.84   -20.24   122.84   395.53  
    ## 
    ## Coefficients:
    ##                               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                  1.093e+01  2.941e-03 3716.91   <2e-16 ***
    ## log(review_cnt + 1)         -4.664e-01  3.263e-03 -142.96   <2e-16 ***
    ## log(downloads + 1)           2.970e-01  3.265e-03   90.95   <2e-16 ***
    ## log(rating + 1)              1.513e-01  3.451e-04  438.43   <2e-16 ***
    ## log(in_degree + 1)          -1.088e-01  4.484e-04 -242.56   <2e-16 ***
    ## log(out_degree + 1)          9.994e-02  5.246e-04  190.50   <2e-16 ***
    ## closeness                   -5.452e+02  7.956e+00  -68.52   <2e-16 ***
    ## log(between + 1)             6.065e-03  1.940e-04   31.26   <2e-16 ***
    ## log(hub_score + 1)           2.868e-01  1.232e-03  232.87   <2e-16 ***
    ## log(authority_score + 1)     8.500e-01  5.135e-03  165.53   <2e-16 ***
    ## log(nghb_mn_review_cnt + 1)  6.363e-02  1.460e-04  435.68   <2e-16 ***
    ## log(nghb_mn_salesrank + 1)   3.367e-02  2.375e-04  141.75   <2e-16 ***
    ## log(nghb_mn_rating + 1)     -7.052e-02  3.978e-04 -177.25   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 16968896  on 517  degrees of freedom
    ## Residual deviance: 14957037  on 505  degrees of freedom
    ## AIC: 14963614
    ## 
    ## Number of Fisher Scoring iterations: 5

The End!
