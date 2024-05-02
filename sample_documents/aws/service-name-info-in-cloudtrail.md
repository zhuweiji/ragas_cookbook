# Amazon EKS information in CloudTrail<a name="service-name-info-in-cloudtrail"></a>

When you create your AWS account, CloudTrail is also enabled on your AWS account\. When any activity occurs in Amazon EKS, that activity is recorded in a CloudTrail event along with other AWS service events in **Event history**\. You can view, search, and download recent events in your AWS account\. For more information, see [Viewing events with CloudTrail event history](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/view-cloudtrail-events.html)\. 

For an ongoing record of events in your AWS account, including events for Amazon EKS, create a trail\. A *trail* enables CloudTrail to deliver log files to an Amazon S3 bucket\. By default, when you create a trail in the console, the trail applies to all AWS Regions\. The trail logs events from all AWS Regions in the AWS partition and delivers the log files to the Amazon S3 bucket that you specify\. Additionally, you can configure other AWS services to further analyze and act upon the event data that's collected in CloudTrail logs\. For more information, see the following resources\.
+ [Overview for creating a trail](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-create-and-update-a-trail.html)
+ [CloudTrail supported services and integrations](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-aws-service-specific-topics.html#cloudtrail-aws-service-specific-topics-integrations)
+ [Configuring Amazon SNS notifications for CloudTrail](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/getting_notifications_top_level.html)
+ [Receiving CloudTrail log files from multiple regions](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/receive-cloudtrail-log-files-from-multiple-regions.html) and [Receiving CloudTrail log files from multiple accounts](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-receive-logs-from-multiple-accounts.html)

All Amazon EKS actions are logged by CloudTrail and are documented in the [Amazon EKS API Reference](https://docs.aws.amazon.com/eks/latest/APIReference/)\. For example, calls to the [https://docs.aws.amazon.com/eks/latest/APIReference/API_CreateCluster.html](https://docs.aws.amazon.com/eks/latest/APIReference/API_CreateCluster.html), [https://docs.aws.amazon.com/eks/latest/APIReference/API_ListClusters.html](https://docs.aws.amazon.com/eks/latest/APIReference/API_ListClusters.html) and [https://docs.aws.amazon.com/eks/latest/APIReference/API_DeleteCluster.html](https://docs.aws.amazon.com/eks/latest/APIReference/API_DeleteCluster.html) sections generate entries in the CloudTrail log files\.

Every event or log entry contains information about the type of IAM identity that made the request, and which credentials were used\. If temporary credentials were used, the entry shows how the credentials were obtained\.

For more information, see the [CloudTrail userIdentity element](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-event-reference-user-identity.html)\.