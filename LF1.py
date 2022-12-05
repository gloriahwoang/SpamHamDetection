import json
import boto3
import email
import numpy
import sms_spam_classifier_utilities as utilities


def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    session = boto3.Session()
    s3_session = session.client('s3')
    response = s3_session.get_object(Bucket=bucket, Key=key)

    email_obj = email.message_from_bytes(response['Body'].read())
    from_email = email_obj.get('From')
    email_receive_date = email_obj['Date']
    
    print ('email_obj:', email_obj)
    print ('email receive date:', email_receive_date)

    
    if type(email_obj.get_payload()) is list:
        body = email_obj.get_payload()[0].get_payload()
    else:
        body = email_obj.get_payload()
    
    
    print('Body: ', body)
    
    print('from_email: ', from_email)
    endpoint_name = 'sms-spam-classifier-mxnet-2022-11-28-16-35-06-049'
    runtime = session.client('runtime.sagemaker')
    vocabulary_length = 9013
    input_mail = [body.strip()]
    print('input_mail1: ', input_mail)
    temp_1 = utilities.one_hot_encode(input_mail, vocabulary_length)
    input_mail = utilities.vectorize_sequences(temp_1, vocabulary_length)
    print('input_mail2: ', input_mail)
    data = json.dumps(input_mail.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=data)
    print('response: ', response)
    res = json.loads(response["Body"].read())

    if res['predicted_label'][0][0] == 0:
        label = 'Ok'
        print('OK')
    else:
        label = 'Spam'
        print('Spam')
    score = round(res['predicted_probability'][0][0], 4)
    score = score*100
   

    message = "We received your email sent at " + email_receive_date + " with the subject " + str(email_obj.get('Subject')) + ".\nHere \
is a 240 character sample of the email body:\n\n" + body[:240] + "\nThe email was \
categorized as " + str(label) + " with a " + str(score) + "% confidence."
    print('message: ', message)
    email_client = session.client('ses')
    response_email = email_client.send_email(
        Destination={'ToAddresses': [from_email]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(email_obj.get('To')),
    )
    print(response_email)
    return {}