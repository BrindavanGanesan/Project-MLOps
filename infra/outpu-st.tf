output "nlb_dns_name" {
  value = aws_lb.iris_nlb.dns_name
}
output "api_invoke_url" {
  value = aws_apigatewayv2_stage.prod.invoke_url
}
