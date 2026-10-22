  Project Summary

This project analyses e-commerce transactions to segment customers, discover products frequently purchased together, and generate personalised product recommendations.
 
 Dataset Summary
- Customers analysed: 1,000
- Products analysed: 4,556
- Transaction events: 528,085
- Customer segments: 4
- Association rules generated: 100

Customer Segments

 Segment -> Customers 
 Loyal Customers -> 434
 Potential Loyalists -> 400 
 Champions -> 99 
 At Risk -> 67 

Customer segmentation uses Recency, Frequency, and Monetary value features followed by K-Means clustering.

Association Rule Mining

The strongest discovered association was:

| Product A | Product B | Support | Confidence | Lift |

| CAST IRON HOOK GARDEN TROWEL | CAST IRON HOOK GARDEN FORK | 0.0041 | 0.8889 | 182.68 |

A high lift means that the two products are purchased together much more often than expected by chance.

For Example Recommendations

For customer `13078.0`, the top recommendation was:

1. WHITE HANGING HEART T-LIGHT HOLDER
2. JUMBO BAG RED WHITE SPOTTY
3. PACK OF 72 RETRO SPOT CAKE CASES

The complete recommendations for all customers are available in `outputs/recommendations.csv`.

Evaluation Results

| Metric | Result |
| Evaluated customers | 1,000 |
| Precision@10 | 0.0012 |
| Recall@10 | 0.0120 |
| Hit Rate@10 | 0.0120 |
| NDCG@10 | 0.004804 |

These results represent the baseline performance of the recommendation model using a time-based holdout evaluation.

All the outputs are saved in the folder # outputs 