resource "aws_ecs_service" "iris_service" {
  name             = "iris-api-service"
  cluster          = aws_ecs_cluster.iris_cluster.id
  launch_type      = "FARGATE"
  desired_count    = 1
  platform_version = "LATEST"

  deployment_controller {
    type = "ECS"
  }

  health_check_grace_period_seconds = 30

  lifecycle {
    ignore_changes = [task_definition]
  }

  task_definition = aws_ecs_task_definition.iris_task.arn

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.iris_nlb_tg.arn
    container_name   = "iris-api"
    container_port   = 8080
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  depends_on = [
    aws_lb_listener.iris_nlb_listener
  ]
}
