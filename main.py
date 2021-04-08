import load_data as l
import data_transformation as t
import create_additional_trx_features as f
import modeling as m

if __name__ == '__main__':
    portfolio = l.load_portfolio('data/portfolio.json')
    profile = l.load_portfolio('data/profile.json')
    transcript = l.load_portfolio('data/transcript.json')

    portfolio = t.transform_portfolio(portfolio)
    profile = t.transform_profile(profile)
    transcript = t.transform_transcript(transcript)
    combine_df = f.create_combined_df (portfolio, profile)

    training_df = combine_df[combine_df['training_label'] == 1]
    testing_df = combine_df[combine_df['training_label'] == 0]

    # Hybrid GBDT (LightGBM) & LR model
    ## Train gbm model for feature transformation
    x_train, x_valid, y_train, y_valid, model = m.create_train_gbm_model(training_df, testing_df)

    ## Train logistic regression with  tree features
    train_ne_hybrid, val_ne_hybrid, test_ne_hybrid, lr_tree_predictions \
    = m.train_logistic_regression_tree_features(model, training_df, testing_df)

    #Logistic regression without tree features
    train_ne_lr, val_ne_lr, test_ne_lr, lr_predictions \
    = m.train_logistic_regression(x_train, x_valid, y_train, y_valid, testing_df)

    #Tree model without tree features
    train_ne_gbm, val_ne_gbm, test_ne_gbm, lgbm_predictions \
    = m.train_lgbm(model, x_train, x_valid, y_train, y_valid, testing_df)

    # Print evaluation metrics
    print('train-ne (hybrid): ', train_ne_hybrid)
    print('val-ne (hybrid): ', val_ne_hybrid)
    print('test-ne (hybrid): ', test_ne_hybrid)

    print('train-ne (LR-only): ', train_ne_lr)
    print('val-ne (LR-only): ', val_ne_lr)
    print('test-ne (LR-only): ', test_ne_lr)

    print('train-ne (LGBM-only): ', train_ne_gbm)
    print('val-ne (LGBM-only): ', val_ne_gbm)
    print('test-ne (LGBM-only): ', test_ne_gbm)

    # save result to csv
    lr_tree_predictions.to_csv('result/lr_tree_predictions.csv')
    lr_predictions.to_csv('result/lr_predictions.csv')
    lgbm_predictions.to_csv('result/lgbm_predictions.csv')
