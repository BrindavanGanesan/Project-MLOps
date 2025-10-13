resource "aws_lb" "iris_lb" {
  name               = "iris-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs_sg.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
}

resource "aws_lb_target_group" "iris_tg" {
  name     = "iris-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  target_type = "ip"
}

resource "aws_lb_listener" "iris_listener" {
  load_balancer_arn = aws_lb.iris_lb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.iris_tg.arn
  }
}

resource "aws_ecs_service" "iris_service" {
  name            = "iris-api-service"
  cluster         = aws_ecs_cluster.iris_cluster.id
  task_definition = aws_ecs_task_definition.iris_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.iris_tg.arn
    container_name   = "iris-api"
    container_port   = 8080
  }

  depends_on = [aws_lb_listener.iris_listener]
}
