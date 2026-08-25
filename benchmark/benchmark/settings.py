# Copyright(C) Facebook, Inc. and its affiliates.
from json import load, JSONDecodeError


class SettingsError(Exception):
    pass


class Settings:
    def __init__(self, key_name, key_path, base_port,client_base_port,client_run_port, repo_name, repo_url,
                 branch, instance_type, aws_regions, use_private_ips=True):
        inputs_str = [
            key_name, key_path, repo_name, repo_url, branch, instance_type
        ]
        if isinstance(aws_regions, list):
            regions = aws_regions
        else:
            regions = [aws_regions]
        inputs_str += regions
        ok = all(isinstance(x, str) for x in inputs_str)
        ok &= isinstance(base_port, int)
        ok &= len(regions) > 0
        if not ok:
            raise SettingsError('Invalid settings types')

        self.key_name = key_name
        self.key_path = key_path

        self.base_port = base_port

        self.client_base_port = client_base_port
        self.client_run_port = client_run_port

        self.repo_name = repo_name
        self.repo_url = repo_url
        self.branch = branch

        self.instance_type = instance_type
        self.aws_regions = regions

        # Address the nodes dial each other on. Private (VPC-internal) addresses
        # keep inter-node traffic inside the region: it never leaves for the
        # internet gateway, so it is not billed as data transfer and the round
        # trip is shorter. Public addresses are still used for SSH.
        #
        # Private addresses are only routable *within* a region, so a
        # multi-region testbed falls back to public ones (see
        # InstanceManager.hosts).
        self.use_private_ips = bool(use_private_ips)

    @classmethod
    def load(cls, filename):
        try:
            with open(filename, 'r') as f:
                data = load(f)

            return cls(
                data['key']['name'],
                data['key']['path'],
                data['port'],
                data['client_base_port'],
                data['client_run_port'],
                data['repo']['name'],
                data['repo']['url'],
                data['repo']['branch'],
                data['instances']['type'],
                data['instances']['regions'],
                data['instances'].get('use_private_ips', True),
            )
        except (OSError, JSONDecodeError) as e:
            raise SettingsError(str(e))

        except KeyError as e:
            raise SettingsError(f'Malformed settings: missing key {e}')
