data "aws_cloudwatch_log_group" "ecs_logs" {
  name = "/ecs/iris-api-task"
}