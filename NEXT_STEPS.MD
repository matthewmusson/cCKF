# Next Steps for this project

## CKF

1. Potentially re-optimize the CKF config on a larger set of parameters and evaluate what kind of config we want to use to collect training data
2. Use a larger set of events to collect training data, determine what n\sigma bounding box for data collection we should use
3. Re-expand all of the events with much tighter audits on the output, i.e. take the hits selected by the CKF and construct data,label pairs from all of the hits the CKF could have chosen at a given layer.
4. Re-propagate the CKF from wrong decisions to get the correct labels for the value function, audit these labels
5. Train both functions and audit the training curves, AUC-ROC, AUC-PR, etc. Here there are a lot of ablations you could run wherein you train on majority defined seeds or just pure seeds. You can also calibrate the output, which may be necessary if you subsample negative examples (which there are much more of in the training set - I got around this by using a very large minibatch).
6. Run EHVI sweeps on the gate and value function to determine at what thresholds you see best efficiency, fake rate, and latency.
7. Run the dataset augmentation loop I outliend with. Here we can sample different operating points from the pareto front, run the CKF with the gate and value function multiple times with each of the (tau_g, tau_v) pairs, then retrain the models on that data + the original CKF data.
8. Try to fully parallelize seeding and the CKF on GPUs...

