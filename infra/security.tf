# Your existing ecs_sg, add/adjust an ingress rule for port 8080
resource "aws_security_group" "ecs_sg" {
  name        = "ecs_sg"
  description = "ECS tasks SG"
  vpc_id      = aws_vpc.main.id

  # Allow 8080 from inside the VPC (NLB nodes live in these subnets)
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block] # 10.0.0.0/16 in your setup
  }

  # Egress to anywhere (for pulling images, S3, etc.)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
