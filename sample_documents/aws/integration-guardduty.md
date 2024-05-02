# Detect threats with Amazon GuardDuty<a name="integration-guardduty"></a>

Amazon GuardDuty is a threat detection service that helps protect you accounts, containers, workloads, and the data with your AWS environment\. Using machine learning \(ML\) models, and anomaly and threat detection capabilities, GuardDuty continuously monitors different log sources and runtime activity to identify and prioritize potential security risks and malicious activities in your environment\.

Among other features, GuardDuty offers the following two features that detect potential threats to your EKS clusters: *EKS Protection* and *Runtime Monitoring*\.

**EKS Protection**  
This feature provides threat detection coverage to help you protect Amazon EKS clusters by monitoring the associated *Kubernetes audit logs*\. Kubernetes audit logs capture sequential actions within your cluster, including activities from users, applications using the Kubernetes API, and the control plane\. For example, GuardDuty can identify that APIs called to potentially tamper with resources in a Kubernetes cluster were invoked by an unauthenticated user\.  
When you enable EKS Protection, GuardDuty will be able to access your Amazon EKS audit logs only for continuous threat detection\. If GuardDuty identifies a potential threat to your cluster, it generates an associated Kubernetes audit log *finding* of a specific type\. For more information about the types of findings available from Kubernetes audit logs, see [Kubernetes audit logs finding types](https://docs.aws.amazon.com/guardduty/latest/ug/guardduty_finding-types-kubernetes.html) in the Amazon GuardDuty User Guide\.  
For more information, see [EKS Protection](https://docs.aws.amazon.com/guardduty/latest/ug/kubernetes-protection.html) in the Amazon GuardDuty User Guide\.

**Runtime Monitoring**  
This feature monitors and analyzes operating system\-level, networking, and file events to help you detect potential threats in specific AWS workloads in your environment\.  
When you enable *Runtime Monitoring* and install the GuardDuty agent in your Amazon EKS clusters, GuardDuty starts monitoring the runtime events associated with this cluster\. If GuardDuty identifies a potential threat to your cluster, it generates an associated *Runtime Monitoring finding*\. For example, a threat can potentially start by compromising a single container that runs a vulnerable web application\. This web application might have access permissions to the underlying containers and workloads\. In this scenario, incorrectly configured credentials could potentially lead to a broader access to the account, and the data stored within it\.  
To configure *Runtime Monitoring*, you install the GuardDuty agent to your cluster as an *Amazon EKS add\-on*\. For more information the add\-on, see [Available Amazon EKS add\-ons from Amazon EKS](eks-add-ons.md#workloads-add-ons-available-eks)\.  
For more information, see [Runtime Monitoring](https://docs.aws.amazon.com/guardduty/latest/ug/runtime-monitoring.html) in the Amazon GuardDuty User Guide\.