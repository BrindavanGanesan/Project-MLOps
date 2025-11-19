resource "aws_lb" "iris_nlb" {
  name               = "iris-nlb"
  load_balancer_type = "network"
  internal           = false
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
}

resource "aws_lb_target_group" "iris_nlb_tg" {
  name        = "iris-nlb-tg"
  port        = 8080
  protocol    = "TCP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    protocol            = "HTTP"
    port                = "8080"
    path                = "/ping"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 15
    timeout             = 5
  }
}

resource "aws_lb_listener" "iris_nlb_listener" {
  load_balancer_arn = aws_lb.iris_nlb.arn
  port              = 80
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.iris_nlb_tg.arn
  }
}
