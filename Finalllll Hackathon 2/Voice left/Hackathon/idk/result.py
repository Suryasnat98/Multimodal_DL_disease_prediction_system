# # 1. Get symptoms (text or voice)
# if use_voice:
#     symptoms_text = speech_to_text(audio_file)
# else:
#     symptoms_text = user_input

# # 2. Predictions
# symptom_preds = predict_symptoms(symptoms_text)
# image_preds = predict_image(img_array)

# # 3. Fusion
# final_preds = fuse_predictions(symptom_preds, image_preds)
# final_preds = normalize(final_preds)

# # 4. Result
# result = get_top_prediction(final_preds)