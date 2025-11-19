#############################################
# Network Load Balancer
#############################################
resource "aws_lb" "iris_nlb" {
  name               = "iris-nlb"
  load_balancer_type = "network"
  internal           = false
  subnets            = [
    aws_subnet.public_a.id,
    aws_subnet.public_b.id
  ]
}


#############################################
# Target Group (TCP → ECS Fargate Tasks)
#############################################
resource "aws_lb_target_group" "iris_nlb_tg" {
  name        = "iris-nlb-tg"
  port        = 8080                  # Your FastAPI container port
  protocol    = "TCP"                 # Must be TCP for NLB
  target_type = "ip"                  # Required for ECS Fargate
  vpc_id      = aws_vpc.main.id

  # Recommended TCP health check
  health_check {
    protocol            = "TCP"
    port                = "8080"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 10
    timeout             = 6
  }
}


#############################################
# Listener (TCP Port 80 → Forward to TG)
#############################################
resource "aws_lb_listener" "iris_nlb_listener" {
  load_balancer_arn = aws_lb.iris_nlb.arn
  port              = 80
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.iris_nlb_tg.arn
  }
}


#############################################
# Output
#############################################
output "nlb_dns_name" {
  value = aws_lb.iris_nlb.dns_name
}
