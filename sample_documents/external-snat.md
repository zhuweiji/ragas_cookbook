# SNAT for Pods<a name="external-snat"></a>

If you deployed your cluster using the `IPv6` family, then the information in this topic isn't applicable to your cluster, because `IPv6` addresses are not network translated\. For more information about using `IPv6` with your cluster, see [`IPv6` addresses for clusters, Pods, and services](cni-ipv6.md)\.

By default, each Pod in your cluster is assigned a [private](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-instance-addressing.html#concepts-private-addresses) `IPv4` address from a classless inter\-domain routing \(CIDR\) block that is associated with the VPC that the Pod is deployed in\. Pods in the same VPC communicate with each other using these private IP addresses as end points\. When a Pod communicates to any `IPv4` address that isn't within a CIDR block that's associated to your VPC, the Amazon VPC CNI plugin \(for both [https://github.com/aws/amazon-vpc-cni-k8s#amazon-vpc-cni-k8s](https://github.com/aws/amazon-vpc-cni-k8s#amazon-vpc-cni-k8s) or [https://github.com/aws/amazon-vpc-cni-plugins/tree/master/plugins/vpc-bridge](https://github.com/aws/amazon-vpc-cni-plugins/tree/master/plugins/vpc-bridge)\) translates the Pod's `IPv4` address to the primary private `IPv4` address of the primary [elastic network interface](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-eni.html#eni-basics) of the node that the Pod is running on, by default [\*](#snat-exception)\.

**Note**  
For Windows nodes, there are additional details to consider\. By default, the [VPC CNI plugin for Windows](https://github.com/aws/amazon-vpc-cni-plugins/tree/master/plugins/vpc-bridge) is defined with a networking configuration in which the traffic to a destination within the same VPC is excluded for SNAT\. This means that internal VPC communication has SNAT disabled and the IP address allocated to a Pod is routable inside the VPC\. But traffic to a destination outside of the VPC has the source Pod IP SNAT'ed to the instance ENI's primary IP address\. This default configuration for Windows ensures that the pod can access networks outside of your VPC in the same way as the host instance\.

Due to this behavior:
+ Your Pods can communicate with internet resources only if the node that they're running on has a [public](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-instance-addressing.html#concepts-public-addresses) or [elastic](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-eips.html) IP address assigned to it and is in a [public subnet](https://docs.aws.amazon.com/vpc/latest/userguide/configure-subnets.html#subnet-basics)\. A public subnet's associated [route table](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Route_Tables.html) has a route to an internet gateway\. We recommend deploying nodes to private subnets, whenever possible\.
+ For versions of the plugin earlier than `1.8.0`, resources that are in networks or VPCs that are connected to your cluster VPC using [VPC peering](https://docs.aws.amazon.com/vpc/latest/peering/what-is-vpc-peering.html), a [transit VPC](https://docs.aws.amazon.com/whitepapers/latest/aws-vpc-connectivity-options/transit-vpc-option.html), or [AWS Direct Connect](https://docs.aws.amazon.com/directconnect/latest/UserGuide/Welcome.html) can't initiate communication to your Pods behind secondary elastic network interfaces\. Your Pods can initiate communication to those resources and receive responses from them, though\.

If either of the following statements are true in your environment, then change the default configuration with the command that follows\.
+ You have resources in networks or VPCs that are connected to your cluster VPC using [VPC peering](https://docs.aws.amazon.com/vpc/latest/peering/what-is-vpc-peering.html), a [transit VPC](https://docs.aws.amazon.com/whitepapers/latest/aws-vpc-connectivity-options/transit-vpc-option.html), or [AWS Direct Connect](https://docs.aws.amazon.com/directconnect/latest/UserGuide/Welcome.html) that need to initiate communication with your Pods using an `IPv4` address and your plugin version is earlier than `1.8.0`\.
+ Your Pods are in a [private subnet](https://docs.aws.amazon.com/vpc/latest/userguide/configure-subnets.html#subnet-basics) and need to communicate outbound to the internet\. The subnet has a route to a [NAT gateway](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html)\.

```
kubectl set env daemonset -n kube-system aws-node AWS_VPC_K8S_CNI_EXTERNALSNAT=true
```

**Note**  
The `AWS_VPC_K8S_CNI_EXTERNALSNAT` and `AWS_VPC_K8S_CNI_EXCLUDE_SNAT_CIDRS` CNI configuration variables aren't applicable to Windows nodes\. Disabling SNAT isn't supported for Windows\. As for excluding a list of `IPv4` CIDRs from SNAT, you can define this by specifying the `ExcludedSnatCIDRs` parameter in the Windows bootstrap script\. For more information on using this parameter, see [Bootstrap script configuration parameters](eks-optimized-windows-ami.md#bootstrap-script-configuration-parameters)\.

 \*If a Pod's spec contains `hostNetwork=true` \(default is `false`\), then its IP address isn't translated to a different address\. This is the case for the `kube-proxy` and Amazon VPC CNI plugin for Kubernetes Pods that run on your cluster, by default\. For these Pods, the IP address is the same as the node's primary IP address, so the Pod's IP address isn't translated\. For more information about a Pod's `hostNetwork` setting, see [PodSpec v1 core](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.29/#podspec-v1-core) in the Kubernetes API reference\. 