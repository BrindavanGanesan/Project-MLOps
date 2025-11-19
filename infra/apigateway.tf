# HTTP API
resource "aws_apigatewayv2_api" "iris_api" {
  name          = "iris-http-api"
  protocol_type = "HTTP"
}

# VPC Link for private integration to the NLB
# Use the same subnets that have network reachability to the NLB (your public subnets are fine here)
resource "aws_apigatewayv2_vpc_link" "iris_vpc_link" {
  name               = "iris-vpc-link"
  subnet_ids         = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  security_group_ids = [aws_security_group.vpc_link_sg.id]
}

# Security group for the VPC Link ENIs (egress allowed)
resource "aws_security_group" "vpc_link_sg" {
  name        = "vpc-link-sg"
  description = "Allows API Gateway VPC Link ENIs to reach NLB"
  vpc_id      = aws_vpc.main.id

  # ENIs need egress to the NLB nodes
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "vpc-link-sg"
  }
}

# Private integration via VPC Link to the NLB LISTENER ARN
resource "aws_apigatewayv2_integration" "iris_integration" {
  api_id                 = aws_apigatewayv2_api.iris_api.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  connection_type        = "VPC_LINK"
  connection_id          = aws_apigatewayv2_vpc_link.iris_vpc_link.id
  integration_uri        = aws_lb_target_group.iris_nlb_tg.arn # NLB listener ARN
  payload_format_version = "1.0"
  timeout_milliseconds   = 29000
}

# ANY /{proxy+} to forward all paths to your service
resource "aws_apigatewayv2_route" "iris_any_proxy" {
  api_id    = aws_apigatewayv2_api.iris_api.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.iris_integration.id}"
}

# Root path too
resource "aws_apigatewayv2_route" "iris_root" {
  api_id    = aws_apigatewayv2_api.iris_api.id
  route_key = "ANY /"
  target    = "integrations/${aws_apigatewayv2_integration.iris_integration.id}"
}

# Stage
resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.iris_api.id
  name        = "prod"
  auto_deploy = true
}

output "api_invoke_url" {
  value = aws_apigatewayv2_stage.prod.invoke_url
}
